import json
import numpy as np
import matplotlib.pyplot as plt

log = open("log/log_008.txt", "r")
lines = log.readlines()
int24_coeff = np.array([256 ** 2, 256, 1]).reshape((1, 3))

n_channels = 30
eeg = np.empty((n_channels, 0))

i = 0
for line in lines:
    # print(json.loads(line)['data']['eeg'][9:])
    # print(len(json.loads(line)['data']['eeg'][9:]))
    eeg_sample = np.array(json.loads(line)['data']['eeg'][9:]).reshape((-1, 3))
    eeg_sample *= int24_coeff
    eeg_sample = np.sum(eeg_sample, axis=1)
    eeg_sample = np.where(eeg_sample > 2 ** 24 / 2, eeg_sample - 2 ** 24, eeg_sample).astype(float)
    eeg_sample *= 20. / 1000.
    eeg_sample = eeg_sample.reshape((10, -1)).T
    eeg = np.hstack([eeg, eeg_sample[:n_channels, :]])

    i += 1
    # if i == 100:
    #     break

from scipy import signal
#
# sos = signal.butter(5, [10, 30], 'band', fs=1000, output='sos')
# for i in range(n_channels):
#     eeg[i] = signal.sosfilt(sos, eeg[i])

# eeg = eeg - eeg[:1, :]
# eeg[6, :] = 0
# eeg *= 5

fig, axs = plt.subplots(4, gridspec_kw={'height_ratios': [1, 1, 1, 7]})
fig.suptitle('Vertically stacked subplots')

axs[3].plot(np.linspace(0, (eeg.shape[1] - 1) / 1000, eeg.shape[1]), eeg[:n_channels].T + 95 * np.arange(n_channels - 1, -1, -1), linewidth=1.0)
axs[3].plot(np.linspace(0, (eeg.shape[1] - 1) / 1000, eeg.shape[1]),
            np.zeros(eeg[:n_channels].T.shape) + 95 * np.arange(n_channels - 1, -1, -1), '--', color='gray')
axs[3].set_xlim(0, (eeg.shape[1] - 1) / 1000)
plt.show()
