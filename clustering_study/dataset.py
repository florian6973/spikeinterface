import spikeinterface.full as si
# import spikeinterface.generate as generate_drifting_recording
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from configuration import base_path

# from spikeinterface.sortingcomponents.tools import remove_empty_templates
# from spikeinterface.core.node_pipeline import sorting_to_peaks
# from spikeinterface.sortingcomponents.benchmark.benchmark_clustering import ClusteringStudy



def create_dataset(dataset_name):

    keys = dataset_name.split('_')

    probe_name = keys[0]
    num_units = int(keys[1])
    duration = keys[2]
    assert duration.endswith('s')
    duration = float(duration[:-1])

    static, drifting, sorting, extra_infos = si.generate_drifting_recording(probe_name=probe_name,
                                                               num_units=num_units,
                                                               duration=300.,
                                                               seed=2205,
                                                               extra_outputs=True,
                                                               )
    # print( static, drifting, sorting)
    return static, drifting, sorting


def get_dataset(dataset_name):
    si.set_global_job_kwargs(n_jobs=0.8)

    dataset_folder = base_path / 'SimulatedDatasetsCache' / dataset_name

    static_rec_folder = dataset_folder / 'static_recording'
    drifting_rec_folder = dataset_folder / 'drifting_recording'
    sorting_folder = dataset_folder / 'sorting_recording'
    static_analyzer_folder =  dataset_folder / 'static_gt_analyzer'
    drifting_analyzer_folder =  dataset_folder / 'drifting_gt_analyzer'


    if not static_rec_folder.exists():
        static, drifting, sorting = create_dataset(dataset_name)

        static_saved = static.save(folder=static_rec_folder)
        drifting_saved = drifting.save(folder=drifting_rec_folder)
        sorting_saved = sorting.save(folder=sorting_folder)
    

        analyzer_static = si.create_sorting_analyzer(sorting_saved, static_saved, sparse=True,
                                                     format="binary_folder", folder=static_analyzer_folder, )
        analyzer_drifting = si.create_sorting_analyzer(sorting_saved, drifting_saved, sparsity=analyzer_static.sparsity,
                                                       format="binary_folder", folder=drifting_analyzer_folder, )
        
        for analyzer in (analyzer_static, analyzer_drifting):
            analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
            analyzer.compute("waveforms")
            analyzer.compute("templates")
            analyzer.compute("noise_levels")
            analyzer.compute("unit_locations")
            analyzer.compute("isi_histograms", window_ms=50., bin_ms=1., method="numba")
            analyzer.compute("correlograms", window_ms=50., bin_ms=1.)
            analyzer.compute("template_similarity")
            analyzer.compute("principal_components", n_components=3, mode='by_channel_global', whiten=True)
            # sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_local', whiten=True, **job_kwargs)
            analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
            analyzer.compute("spike_amplitudes")


    else:
        static_saved = si.load_extractor(static_rec_folder)
        drifting_saved = si.load_extractor(drifting_rec_folder)
        sorting_saved = si.load_extractor(sorting_folder)
        analyzer_static = si.load_sorting_analyzer(static_analyzer_folder)
        analyzer_drifting = si.load_sorting_analyzer(drifting_analyzer_folder)

    return static_saved, drifting_saved, sorting_saved, analyzer_static, analyzer_drifting

def open_sigui(dataset_name, static=True):

    static_saved, drifting_saved, sorting_saved, analyzer_static, analyzer_drifting = get_dataset(dataset_name)

    if static:
        analyzer = analyzer_static
    else:
        analyzer = analyzer_drifting

    si.plot_sorting_summary(analyzer, backend="spikeinterface_gui")

    

if __name__ == '__main__':
    #dataset_name = 'Neuronexus-32_50_300.s'
    dataset_name = 'Neuropixel-128_250_300.s'
    # dataset_name = 'Neuropixel-128_800_300.s'

    create_dataset(dataset_name)

    # static_saved, drifting_saved, sorting_saved, analyzer_static, analyzer_drifting = get_dataset(dataset_name)
    # print(analyzer_static)

    open_sigui(dataset_name)
