import fcwt
import numpy as np
import matplotlib.pyplot as plt

# Initialize
fs = 1000
n = fs * 1  # 100 seconds
ts = np.arange(n)

# Generate linear chirp
signal = np.sin(2 * np.pi * ((1 + (20 * ts) / n) * (ts / fs)))

f0 = 1  # lowest frequency
f1 = 41  # highest frequency
fn = 80  # number of frequencies

#Calculate CWT without plotting...
import time
start = time.time()
freqs, out = fcwt.cwt(signal, fs, f0, f1, fn)
print(time.time() - start)

plt.imshow(out.real)
plt.show()