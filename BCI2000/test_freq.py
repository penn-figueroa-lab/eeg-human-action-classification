import fcwt
import mne
import numpy as np
from mne.datasets import eegbci
import sklearn.model_selection
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import time
import threading

subj = 1
runs = 7
tmin = -0.2
tmax = 1
event_ids = dict(left=0, right=1)

raw_fnames = eegbci.load_data(subj, runs, update_path=True)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.io.concatenate_raws(raws)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
# extract events
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=0, T2=1))
# create epochs
epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)

print(epochs.get_data()[:,0,:].shape)
# plt.plot(epochs.get_data()[0,0,:])

########## test first epoch channal 1 ############
test1 = epochs.get_data()[0,0,:]
fs = 160
f0 = 5 #lowest frequency
f1 = 30 #highest frequency
fn = 50 #number of frequencies

start = time.time()
for i in range(64):
    test1 = epochs.get_data()[0,i,:]
    freqs, out = fcwt.cwt(test1, fs, f0, f1, fn)
print(time.time() - start)
############## time ##################
#### for 1 single channal: 0.001s ####
#### for all 64 channals: 0.0126s ####
# fcwt.plot(test1, fs, f0=f0, f1=f1, fn=fn)


########### test multithreading ##########
def func(i):
    test1 = epochs.get_data()[0,i,:]
    freqs, out = fcwt.cwt(test1, fs, f0, f1, fn)

start = time.time()
for i in range(64):
    t = threading.Thread(target=func, args=(i,))
    t.start()

for thread in threading.enumerate():
    if thread != threading.current_thread():
        thread.join()
print(time.time() - start)
###########################################

########### Concate all channels together #########
print(out.real.shape)
plt.imshow(out.real)
plt.show()