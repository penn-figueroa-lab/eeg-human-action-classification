import fcwt
import numpy as np
import matplotlib.pyplot as plt

# Initialize
fs = 1000
n = fs * 1  # 100 seconds
ts = np.arange(n)

# Generate linear chirp
signal = np.sin(2 * np.pi * ((1 + (20 * ts) / n) * (ts / fs)))

f0 = .1  # lowest frequency
f1 = 201  # highest frequency
fn = 500  # number of frequencies

#Calculate CWT without plotting...
freqs, out = fcwt.cwt(signal, fs, f0, f1, fn)

plt.imshow(out.real)
plt.show()