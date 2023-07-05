import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

np.random.seed(22)

# Number of samples
ns = np.linspace(0, 200, 1000)

# Source matrix
S = np.array([np.sin(ns * 1),
              signal.sawtooth(ns * 1.9),
              np.random.random(len(ns))]).T

# Mixing matrix
A = np.array([[0.5, 1, 0.2],
              [1, 0.5, 0.4],
              [0.5, 0.8, 1]])

# Mixed signal matrix
X = S.dot(A).T

from ica import ICA

n_components = 3
filter = ICA(n_components)
unMixed = filter.fit(X).T
filter.exclude = [2]
Filtered = filter.apply(X).T

fig, axs = plt.subplots(4, 1, figsize=[18, 20])
axs[0].plot(S, lw=5)
axs[0].tick_params(labelsize=12)
axs[0].set_yticks([-1, 1])
axs[0].set_title(r'Source signals ($S$)')
axs[0].set_xlim(0, 100)

axs[1].plot(X.T, lw=5)
axs[1].tick_params(labelsize=12)
axs[1].set_yticks([-1, 1])
axs[1].set_title(r'Mixed signals ($X$)')
axs[1].set_xlim(0, 100)

axs[2].plot(unMixed, lw=5)
axs[2].tick_params(labelsize=12)
axs[2].set_yticks([-1, 1])
axs[2].set_title(r'Unmixed signals ($WX$)')
axs[2].set_xlim(0, 100)

axs[3].plot(Filtered, lw=5)
axs[3].tick_params(labelsize=12)
axs[3].set_yticks([-1, 1])
axs[3].set_title(r'Filtered signals ($W^{-1}S$)')
axs[3].set_xlim(0, 100)

fig.suptitle('ICA results', fontsize=30)
plt.show()
