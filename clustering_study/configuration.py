from pathlib import Path
import getpass
import sys

import spikeinterface.full as si

if getpass.getuser() == 'samuel' and sys.platform.startswith('linux'):
    base_path = Path('/home/samuel/DataSpikeSorting/data_study_components_edimbourgh')
    lussac_data_path = None
    kilosort4_data_path = None
    kilosort2_5_path = None

elif getpass.getuser() == 'samuel.garcia' and sys.platform.startswith('linux'):
    base_path = Path('/mnt/data/sam/DataSpikeSorting/data_study_components_edimbourgh')
    lussac_data_path = Path('/mnt/data/sam/DataSpikeSorting/lussac_dataset')
    kilosort4_data_path = Path('/mnt/data/sam/DataSpikeSorting/DataKilosort4/')
    kilosort2_5_path = '/home/samuel.garcia/Documents/SpikeInterface/code_sorters/Kilosort2.5/'

elif getpass.getuser() == 'pierre' and sys.platform.startswith('linux'):
    base_path = Path('/home/pierre/edimbourgh_laptop')
    lussac_data_path = Path('/media/pierre/Transcend/lussac/')
    kilosort4_data_path = None
    kilosort2_5_path = None

elif getpass.getuser() == 'cure' and sys.platform.startswith('linux'):
    base_path = Path('/home/cure/edimbourgh')
    lussac_data_path = None
    kilosort4_data_path = None
    kilosort2_5_path = None

elif getpass.getuser() == 'axoft_soma' and sys.platform.startswith('linux'):
    base_path = Path('/home/axoft_soma/_code/si_flo/clustering_study/data')
    lussac_data_path = None
    kilosort4_data_path = None
    kilosort2_5_path = None