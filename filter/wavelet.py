import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy.experimental import scale_to_freq


# %%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

plt.plot(xo)
plt.show()
plt.plot(x)
plt.show()

# %%# With units #######################################
from ssqueezepy import Wavelet, cwt, imshow, icwt

fs = 400
t = np.linspace(0, N / fs, N)
wavelet = Wavelet()
Wx, scales = cwt(x, wavelet)

freqs_cwt = scale_to_freq(scales, wavelet, len(x), fs=fs)

ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
imshow(Wx, **ikw, yticks=freqs_cwt)
