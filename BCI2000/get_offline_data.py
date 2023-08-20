import mne
import numpy as np
from mne.datasets import eegbci
import sklearn.model_selection
from tqdm import tqdm
import random

"""Load dataset which mixing all subjects trails"""
def loadDataset(train_size=0.9, tmin=-2, tmax=4):
    np.random.seed(32)
    all_subj = np.arange(1,110)
    all_data = []
    all_label = []
    for subj in tqdm(all_subj):
        # left, right
        event_ids = dict(rest=0, left=1, right=2)
        for runs in [3,7,11]:
            raw_fnames = eegbci.load_data(subj, runs, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.io.concatenate_raws(raws)
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            # extract events
            events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
            events = events[1:, :]
            # create epochs
            epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)
            if epochs.get_data().shape[2] != 961:
                print(">>>>>>>>>>>", subj)
                continue
            all_data.extend(epochs.get_data())
            all_label.extend(events[:,2])

        # fists, feet
        event_ids = dict(rest=0, fists=3, feet=4)
        for runs in [5,9,13]:
            raw_fnames = eegbci.load_data(subj, runs, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.io.concatenate_raws(raws)
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            # extract events
            events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=3, T2=4))
            events = events[1:, :]
            # create epochs
            epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)
            
            if epochs.get_data().shape[2] != 961:
                print(">>>>>>>>>>>", subj)
                continue
            all_data.extend(epochs.get_data())
            all_label.extend(events[:,2])
        # all_data = np.array(all_data)
        # all_label = np.array(all_label)
        # print(all_data.shape, all_label.shape)

    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print(all_data.shape, all_label.shape)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data, all_label, train_size=train_size, random_state=32, shuffle=True)
    
    return X_train, X_test, y_train, y_test


'''different subjects for train and test'''
def loadDataset1(train_size=0.9, tmin=-2, tmax=4):
    np.random.seed(32)
    random.seed(32)
    all_subj = np.arange(1,110)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train_subj = np.random.choice(all_subj, size=int(109*train_size), replace=False)
    test_subj = np.setdiff1d(all_subj, train_subj)
    print(train_subj, test_subj)
    for subj in tqdm(all_subj):
        event_ids = dict(rest=0, left=1, right=2)
        for runs in [3,7,11]:
            raw_fnames = eegbci.load_data(subj, runs, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.io.concatenate_raws(raws)
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            # extract events
            events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
            # create epochs
            epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)
            if epochs.get_data().shape[2] != 961:
                print(">>>>>>>>>>>", subj)
                continue
            if subj in train_subj:
                X_train.extend(epochs.get_data())
                y_train.extend(events[:,2])
            else:
                X_test.extend(epochs.get_data())
                y_test.extend(events[:,2])
            
        event_ids = dict(rest=0, fists=3, feet=4)
        for runs in [5,9,13]:
            raw_fnames = eegbci.load_data(subj, runs, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.io.concatenate_raws(raws)
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            # extract events
            events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=3, T2=4))
            # create epochs
            epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)
            if epochs.get_data().shape[2] != 961:
                print(">>>>>>>>>>>", subj)
                continue
            if subj in train_subj:
                X_train.extend(epochs.get_data())
                y_train.extend(events[:,2])
            else:
                X_test.extend(epochs.get_data())
                y_test.extend(events[:,2])

    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train, y_train = zip(*combined)
    random.seed(32)
    combined = list(zip(X_test, y_test))
    random.shuffle(combined)
    X_test, y_test = zip(*combined)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


"""To use the functions"""
# X_train, X_test, y_train, y_test = loadDataset()
# X_train_, X_test_, y_train_, y_test_ = loadDataset1()
