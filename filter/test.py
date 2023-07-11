import numpy as np
import matplotlib.pyplot as plt
import spkit as sp

signal, _ = sp.load_data.eegSample()
signal = signal[:, 1]
fs = 128
ts = np.arange(2048)

XR = sp.eeg.ATAR(signal.copy(), verbose=0, beta=0.1, OptMode='soft')
plt.figure(figsize=(12, 5))
plt.plot(ts, signal)
plt.plot(ts, XR)

plt.show()
