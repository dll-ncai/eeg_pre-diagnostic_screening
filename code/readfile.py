
import logging
import re
import numpy as np
import glob
import os.path
import torch#for testing gpu 

import mne



def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration

cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        '0000001.edf')
print(cnt)
print(sfreq)
print( chan_names)
print( n_samples)
print(n_channels)
print(n_sec)
cnt.load_data()
selected_ch_names = []

wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
	        'FP2', 'FZ', 'O1', 'O2',
	        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

for wanted_part in wanted_elecs:
    wanted_found_name = []
    for ch_name in cnt.ch_names:#if ' ' + wanted_part + '-' in ch_name:
        if  wanted_part  in ch_name:
            wanted_found_name.append(ch_name)
    assert len(wanted_found_name) == 1
    selected_ch_names.append(wanted_found_name[0])
#edf_file = mne.io.read_raw_edf('0000014.edf',montage=None, eog=[ 'FP1', 'FP2','F3', 'F4',
#			'C3', 'C4',  'P3', 'P4','O1', 'O2','F7', 'F8',
#	         	'T3', 'T4', 'T5', 'T6','PZ','FZ', 'CZ','A1', 'A2'], verbose='error')


