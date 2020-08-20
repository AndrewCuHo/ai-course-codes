import pickle
from time import time

import numpy as np
import pandas as pd
import scipy
from mne.io import read_raw_edf
from scipy.signal import welch

'''
data: https://physionet.org/content/sleep-edfx/1.0.0/
EOG:100HZ, EEF:100HZ, EMG:1HZ
delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz)
author: 112020333002191
date: 20/8/18
reference: https://raphaelvallat.com/
reference: https://github.com/fjjason/
'''


def bandpower(seq, sampling_frequency, frequency_band, window_sec=None):
    low, high = frequency_band
    nperseg = (2 / low) * sampling_frequency
    # f, Pxx = scipy.signal.periodogram(seq, fs=sampling_frequency)
    f, Pxx = welch(seq, sampling_frequency, nperseg=nperseg)
    ind_min = scipy.argmax(f > frequency_band[0]) - 1
    ind_max = scipy.argmax(f > frequency_band[1]) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


feature_list = []
label_list = []
path_list = []
sf = 100
time_interval = 1200 * sf
base_path = 'F:/EEG_data/SLEEP_EDF/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/'
eeg_info_df = pd.read_csv('./eeg_info_df.csv')
frequency_bands = {'Beta': [12, 30],
                   'Alpha': [8, 12],
                   'Theta': [4, 8],
                   'Delta': [0.5, 4]}
band_names = list(frequency_bands.keys())
shared_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
cnt1 = time()
for i, path in enumerate(eeg_info_df.path.values):
    raw_edf = read_raw_edf(eeg_info_df.path.iloc[i], preload=True)
    edf_df = raw_edf.to_data_frame()
    edf_df = edf_df[abs(edf_df['EEG Fpz-Cz']) < 1000]  # remove outliers
    n = edf_df.shape[0]
    n_hours = n // time_interval
    for j in range(n_hours):
        temp_df = edf_df.iloc[(j * time_interval):((j + 1) * time_interval), :]
        features = [bandpower(temp_df[channel].values, sf, frequency_bands[band]) for channel in shared_channels for
                    band in
                    band_names]
        feature_list.append(features)
        label_list.append(1 if path.find('telemetry') > -1 else 0)
        path_list.append(path)
print('20 min band power calculation for 4 bands x 4 channels takes {} seconds'.format(round(time() - cnt1)))
y = np.array(label_list)
X = np.array(feature_list)
paths = np.array(path_list)

pickle.dump((X, y, paths), open('./20min_data.npy', 'wb'))
