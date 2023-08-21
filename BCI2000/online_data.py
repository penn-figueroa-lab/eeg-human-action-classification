import numpy as np
import random
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
rest_train = np.load('rest_train.npy')
rest_test = np.load('rest_test.npy')
# Whole function
def getDataset(X_train, X_test, y_train, y_test, rest_train, rest_test):
    # Define our parameters
    sampling_freq = 160
    window_size = int(0.5 * sampling_freq)
    step_size = int(0.05 * sampling_freq)
    RP_t = 0.6
    # RP appears 200ms before movement
    motion_t = 0.8
    stop_t = motion_t + 0.5
    RP_d = int(RP_t * sampling_freq)
    stop_d = int((stop_t+0.5) * sampling_freq)
    X_train_new = []
    y_train_new = []
    X_test_new = []
    y_test_new = []
    print("here")
    for j, sample in enumerate(X_train):
        i = 0
        curr_window = sample[:, 0:window_size]
        while i + window_size < stop_d:
            X_train_new.append(curr_window)
            
            curr_window = sample[:, i:(window_size+i)]
            if i + window_size < RP_d:
                y_train_new.append(0)
            else:
                y_train_new.append(y_train[j]+1)
            i += step_size
    for j, sample in enumerate(X_test):
        i = 0
        curr_window = sample[:, 0:window_size]
        while i + window_size < stop_d:
            X_test_new.append(curr_window)
            
            curr_window = sample[:, i:(window_size+i)]
            if i + window_size < RP_d:
                y_test_new.append(0)
            else:
                y_test_new.append(y_test[j]+1)
            i += step_size
    # print(len(X_train_new), X_train_new[0].shape)
    X_train_new += rest_train.tolist()
    y_train_new += np.zeros((rest_train.shape[0],)).tolist()
    X_test_new += rest_test.tolist()
    y_test_new += np.zeros((rest_test.shape[0],)).tolist()
    random.seed(32)
    combined = list(zip(X_train_new, y_train_new))
    random.shuffle(combined)
    X_train, y_train = zip(*combined)
    combined = list(zip(X_test_new, y_test_new))
    random.shuffle(combined)
    X_test, y_test = zip(*combined)
    X_train_new = np.array(X_train)
    y_train_new = np.array(y_train)
    X_test_new = np.array(X_test)
    print('here1')
    y_test_new = np.array(y_test)
    val_size = int(0.5*len(X_test_new))
    return X_train_new , y_train_new, X_test_new[:val_size], y_test_new[:val_size], X_test_new[val_size:], y_test_new[val_size:]
a,b,c,d,e,f = getDataset(X_train, X_test, y_train, y_test, rest_train, rest_test)
#a = getDataset(X_train, X_test, y_train, y_test, rest_train, rest_test)
print(a.shape,b.shape,c.shape,d.shape,e.shape,f.shape)
#np.save('x_train_win.npy', a)
# np.save('y_train_win.npy', b)
# np.save('x_val_win.npy', c)
# np.save('y_val_win.npy', d)
# np.save('x_test_win.npy', e)
# np.save('y_test_win.npy', f)
import numpy as np
import pickle

data_to_save = {
    'x_train': a,
    'y_train': b,
    'x_val': c,
    'y_val': d,
    'x_test': e,
    'y_test': f,
}

# Save the data to a file using pickle
with open('data_file_windowed.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

