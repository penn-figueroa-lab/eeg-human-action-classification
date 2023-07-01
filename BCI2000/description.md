Link to the dataset: https://drive.google.com/drive/u/0/folders/1gsYhX_mTvouvDu_LpJqfnFjEHti1cSl7

Notice: There are 3 subjects (88, 92, 100) whose sampling frequency are not 160Hz, so I just delete them.

Total: 109 - 3 = 106 subjects

X_train: (2866, 64, 561)
- 2866: Number of trails.
- 64: Number of channels.
- 561: Time steps. [-0.5s, 3s] -> 3.5s * 160Hz = 560

X_test: (1911, 64, 561)

y_train: (2866,)
- label 0: left grasping
- label 1: right grapsing

y_test: (1911,)
