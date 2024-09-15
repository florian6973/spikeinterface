import spikeinterface.full as si
from pathlib import Path
from spikeinterface.sortingcomponents.tools import remove_empty_templates
import numpy as np

from configuration import base_path
from slurm_tools import push_to_slurm
from dataset import get_dataset

def preprocess(rec):
    rec_f = si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')
    rec_f = si.common_reference(rec_f, reference='global', operator='median')
    return rec_f


def run_study(study_folder, dataset_name, erase=True):
    si.set_global_job_kwargs(n_jobs=1)

    rng = np.random.default_rng(seed=2205)
    
    static, drifting, sorting, analyzer_static, analyzer_drifting = get_dataset(dataset_name)

    non_whithen = preprocess(static)
    whiten_global = si.whiten(non_whithen, mode="global")
    whiten_local = si.whiten(non_whithen, mode="local", radius_um=100.)
    
    study_folder = Path(study_folder)
    
    datasets = {
        "non-whiten": (non_whithen, sorting),
        "whiten_global" : (whiten_global, sorting),
        "whiten_local" : (whiten_local, sorting),
    }

    
    nb_spikes = sorting.to_spike_vector().size
    max_spikes = int(5000*static.get_num_channels())
    if nb_spikes < max_spikes:
        indices = np.arange(nb_spikes)
    else:
        indices = np.random.choice(np.arange(nb_spikes), min(nb_spikes, max_spikes), replace=False)
    indices = np.sort(indices)
    peaks = {}
    for dataset in datasets.keys():

        recording, gt_sorting = datasets[dataset]
        
        sorting_analyzer = si.create_sorting_analyzer(gt_sorting, recording, format="memory", sparse=False)
        sorting_analyzer.compute(["random_spikes", "templates"])
        from spikeinterface.core.template_tools import get_template_extremum_channel
        extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")
        spikes = gt_sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        peaks[dataset] = spikes
    
    cases = {}
    
    for engine in ['tdc_clustering', 'circus', 'random_projections']:
        for key in datasets.keys():
            cases[(engine, key)] = {
                "label": f"{engine} {key}",
                "dataset": key,
                "init_kwargs": {'indices' : indices, 'peaks' : peaks[key]},
                "params" : {"method" : engine, "method_kwargs" : {}},
            }
    
    
    from spikeinterface.sortingcomponents.benchmark.benchmark_clustering import ClusteringStudy
    if erase:
        import shutil
        if study_folder.exists():
            shutil.rmtree(study_folder)
        study = ClusteringStudy.create(study_folder, datasets=datasets, cases=cases)
    else:
        study = ClusteringStudy(study_folder)
    
    study.run()
    study.compute_results()
    study.compute_metrics()

if __name__ == "__main__":
    global_name = 'clustering_whitening'

    # dataset_name = 'Neuronexus-32_50_300.s'
    dataset_name = 'Neuropixel-128_250_300.s'

    motion_folder = base_path / global_name / dataset_name / 'motion'
    study_folder = base_path / global_name / dataset_name / 'study'
    run_study(study_folder, dataset_name, erase = True)


    # push_to_slurm(run_study,  study_folder,  dataset_name, erase=True,
    #               slurm_option={'mem': '90G', 'cpus-per-task': 70, 'partition': 'GPU'},
    #               block_mode=False,
    #               )

