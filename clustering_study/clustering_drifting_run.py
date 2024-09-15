import spikeinterface.full as si
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



from spikeinterface.sortingcomponents.tools import remove_empty_templates
from spikeinterface.core.node_pipeline import sorting_to_peaks
from spikeinterface.sortingcomponents.benchmark.benchmark_clustering import ClusteringStudy

from configuration import base_path
# from slurm_tools import push_to_slurm
from dataset import get_dataset


def preprocess(rec):
    rec_f = si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')
    rec_f = si.common_reference(rec_f, reference='global', operator='median')
    return rec_f

def run_study(motion_folder, study_folder, dataset_name='Neuronexus-32_50_300.s', erase=True):
    si.set_global_job_kwargs(n_jobs=0.8)
    rng = np.random.default_rng(seed=2205)

    motion_folder = Path(motion_folder)
    study_folder = Path(study_folder)

    static, drifting, sorting, analyzer_static, analyzer_drifting = get_dataset(dataset_name)
    
    static = preprocess(static)
    drifting = preprocess(drifting)
    corrected = si.correct_motion(drifting, folder=motion_folder, preset='nonrigid_fast_and_accurate', overwrite=True)
    
    static = si.whiten(static) # , regularize=True
    corrected = si.whiten(corrected) # , regularize=True
    
    
    datasets = {
       "static" : (static, sorting),
       "corrected" : (corrected, sorting),
    }
    
    nb_spikes = sorting.to_spike_vector().size
    max_spikes = int(5000*static.get_num_channels())
    if nb_spikes < max_spikes:
        indices = np.arange(nb_spikes)
    else:
        indices = rng.choice(np.arange(nb_spikes), min(nb_spikes, max_spikes), replace=False)
    indices = np.sort(indices)
    all_peaks = {}
    for dataset_name in datasets.keys():

        recording, gt_sorting = datasets[dataset_name]
        
        sorting_analyzer = si.create_sorting_analyzer(gt_sorting, recording, format="memory", sparse=False)
        sorting_analyzer.compute(["random_spikes", "templates"])
        sorting_analyzer.compute(["spike_amplitudes"])
        from spikeinterface.core.template_tools import get_template_extremum_channel
        extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")

        # spikes = gt_sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        peaks = sorting_to_peaks(gt_sorting, extremum_channel_inds)
        peaks["amplitude"] = sorting_analyzer.get_extension("spike_amplitudes").get_data()
        all_peaks[dataset_name] = peaks
    
    cases = {}
    
    for engine in ["circus_umap_gpu", 'tdc_clustering', 'circus', 'random_projections']:
    # for engine in ['tdc_clustering', 'circus', ]:
    #for engine in ['circus']:
        for dataset_name in datasets.keys():
            if engine.endswith("cpu") or engine.endswith("gpu"):
                cases[(engine, dataset_name)] = {
                    "label": f"{engine} {dataset_name}",
                    "dataset": dataset_name,
                    "init_kwargs": {'indices' : indices, 'peaks' : all_peaks[dataset_name]},
                    "params": {"method": engine[:-4], "method_kwargs": {
                        "device": "cpu" if engine.endswith("cpu") else "gpu"
                    }},
                }
            else:
                cases[(engine, dataset_name)] = {
                    "label": f"{engine} {dataset_name}",
                    "dataset": dataset_name,
                    "init_kwargs": {'indices' : indices, 'peaks' : all_peaks[dataset_name]},
                    "params" : {"method" : engine, "method_kwargs" : {}},
                }

    # for method in ["circus_umap_gpu", "random_projections", "circus", "tdc_clustering"]:
    # # for method in ["circus_umap_cpu", "circus_umap_gpu", "random_projections", "circus", "tdc_clustering"]:
    #     if method.endswith("cpu") or method.endswith("gpu"):
    #         cases[method] = {
    #             "label": f"{method} on toy",
    #             "dataset": "toy",
    #             "init_kwargs": {"indices": spike_indices, "peaks": peaks["toy"]},
    #             "params": {"method": method[:-4], "method_kwargs": {
    #                 "device": "cpu" if method.endswith("cpu") else "gpu"
    #             }},
    #         }
    #     else:
    #         cases[method] = {
    #             "label": f"{method} on toy",
    #             "dataset": "toy",
    #             "init_kwargs": {"indices": spike_indices, "peaks": peaks["toy"]},
    #             "params": {"method": method, "method_kwargs": {}},
    #         }

    
    if erase:
        import shutil
        if study_folder.exists():
            shutil.rmtree(study_folder)
        study = ClusteringStudy.create(study_folder, datasets=datasets, cases=cases)
    else:
        study = ClusteringStudy(study_folder)
    
    study.run(verbose=True)
    
    study.compute_results()
    study.compute_metrics()



if __name__ == "__main__":
    global_name = 'clustering_drifting'

    # dataset_name = 'Neuronexus-32_50_300.s'
    dataset_name = 'Neuropixel-128_250_300.s'

    motion_folder = base_path / global_name / dataset_name / 'motion'
    study_folder = base_path / global_name / dataset_name / 'study'

    run_study(motion_folder, study_folder, dataset_name, erase = True)


    # push_to_slurm(run_study,  motion_folder, study_folder,  dataset_name, erase=True,
    #               slurm_option={'mem': '90G', 'cpus-per-task': 70, 'partition': 'GPU'},
    #               block_mode=False,
    #               )

    # study = ClusteringStudy(study_folder)
    # print(study)
    # study.plot_performances_vs_snr()
    # study.plot_agreements()
    # plt.show()


    # si.set_global_job_kwargs(n_jobs=-1)
    # study = ClusteringStudy(study_folder)
    # case_keys = [('tdc_clustering', 'static'), ('tdc_clustering', 'corrected')]
    # study.run(case_keys=case_keys, keep=False)
    # study.compute_results(case_keys=case_keys)

    # print(study)
    # study.plot_agreements(case_keys=case_keys)
    # study.plot_metrics_vs_snr(case_keys=case_keys, metric='cosine')
    # study.plot_metrics_vs_snr(case_keys=case_keys, metric='l2')
    # plt.show()

