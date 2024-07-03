import copy
import yaml

import scipy
import numpy as np

import pyriemann

with open('../data/optitrack_onset_motion.yaml', 'r') as file:
    onset_motion_data = yaml.safe_load(file)


def input_size_check(x: np.ndarray):
    n = len(x.shape)
    if n != 3 and n != 2:
        raise Exception('Unsupported input shape!')


def load_eeg_data(filename: str):
    # Load the data
    if filename not in ['data001.npz', 'data002.npz', 'data_opti_101.npz', 'data_opti_102.npz']:
        raise Exception('Wrong dataset name: ' + str(filename))
    try:
        data = np.load(filename)
    except:
        raise Exception('Cannot load the dataset: ' + str(filename))

    # Read the data for training and evaluation
    X = data['X']
    y = data['y']
    cov = data['cov']
    info = {}

    # Read Optitrack action label data if exists
    try:
        info['lopti'] = data['lopti']
        info['ropti'] = data['ropti']
        print('Loaded the data with label: ' + str(filename))
    finally:
        print('Loaded the data: ' + str(filename))

    return X, y, cov, info


def derivative2origianl(x: np.ndarray):
    # x: T x N (if batch=False) or B x T x N (if batch=True)
    # B: batch size, T: time step, N: number of electrodes
    input_size_check(x)
    return x.cumsum(axis=-2)


def sequence2covariance(x: np.ndarray):
    # x: T x N (if batch=False) or B x T x N (if batch=True)
    # B: batch size, T: time step, N: number of electrodes
    input_size_check(x)
    return x.swapaxes(-1, -2) @ x / (x.shape[-2] - 1)


def convariance2vector(x: np.ndarray):
    # x: N x N (if batch=False) or B x N x N (if batch=True)
    # B: batch size, N: number of electrodes
    input_size_check(x)
    N = x.shape[-1]
    index_x, index_y = np.triu_indices(N)
    return x[..., index_x, index_y]


def covariance2tangentspace(x: np.ndarray):
    # x: B x N x N
    # B: batch size, N: number of electrodes
    input_size_check(x)
    mean = pyriemann.utils.mean.mean_covariance(x)  # N x N
    tangentspace = pyriemann.utils.tangentspace.tangent_space(x, mean)  # B x N x N
    return tangentspace, mean


def change_window_size(x: np.ndarray, window_size: int = 320):
    # x: B x N x N
    # B: batch size, N: number of electrodes
    if window_size > 320:
        print('too big window size')
    elif window_size <= 160:
        return x[:, x.shape[1] - window_size:, :]
    x_orig = copy.deepcopy(x)
    x_prev = copy.deepcopy(x)
    x_prev[window_size:] = x_prev[:-window_size]
    x_prev[:window_size] = x_prev[:window_size][:, ::-1, :]
    x_new = np.concatenate([x_prev, x_orig], axis=1)
    return x_new[:, x_new.shape[1] - window_size:, :]


def freq_filter(x: np.ndarray, freq=[0, 10]):
    # x: T x N (if batch=False) or B x T x N (if batch=True)
    # B: batch size, T: time step, N: number of electrodes
    input_size_check(x)
    if freq[0] == 0:
        b, a = scipy.signal.iirfilter(4, Wn=freq[1], fs=160, btype="lowpass", ftype="butter")
    else:
        b, a = scipy.signal.iirfilter(4, Wn=freq, fs=160, btype="bandpass", ftype="butter")

    if len(x.shape) > 2:
        x = [scipy.signal.lfilter(b, a, xi) for xi in x]
        return np.stack(x)
    else:
        return scipy.signal.lfilter(b, a, x)


def clip_onset_motion(x: np.ndarray, filename: str, before_sec: float = 1., after_sec: float = 1.):
    onsets, motions = onset_motion_data[filename]['timing'], onset_motion_data[filename]['label']
    clips_x = []
    clips_y = []
    for t, y in zip(onsets, motions):
        t_min = max(0, t - int(before_sec * 160))
        t_max = min(x.shape[0], t + int(after_sec * 160))
        clips_x.append(x[t_min:t_max])
        clips_y.append(np.ones(clips_x[-1].shape[0]) * y)
    clips_x = np.concatenate(clips_x, axis=0)
    clips_y = np.concatenate(clips_y, axis=0)
    return clips_x, clips_y
