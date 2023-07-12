import fcwt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

train_data = np.load('./data_lr_mix/X_train.npy')
test_data = np.load('./data_lr_mix/X_test.npy')

print(train_data.shape)

# freqs = np.zeros((train_data.shape[0], train_data.shape[1], 25, train_data.shape[2]))
# for trail in tqdm(range(train_data.shape[0])):
#     for channal in range(train_data.shape[1]):
#         data = train_data[trail, channal, :]
#         _, out = fcwt.cwt(data, 160, 5, 30, 25)
#         freqs[trail, channal, :, :] = out.real

# np.save("freq_train.npy", freqs)

freqs = np.zeros((test_data.shape[0], test_data.shape[1], 25, test_data.shape[2]))
for trail in tqdm(range(test_data.shape[0])):
    for channal in range(test_data.shape[1]):
        data = test_data[trail, channal, :]
        _, out = fcwt.cwt(data, 160, 5, 30, 25)
        freqs[trail, channal, :, :] = out.real

np.save("freq_test.npy", freqs)
plt.imshow(freqs[4,12,:,:])
plt.show()