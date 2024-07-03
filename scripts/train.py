from functools import partial
import pyriemann

from utils import load_eeg_data, change_window_size, derivative2origianl, clip_onset_motion, freq_filter, \
    sequence2covariance, covariance2tangentspace, convariance2vector


def get_data(filename: str, window_size: int = 160, diff: bool = True,
             freq=None, clip: bool = True, method: str = 'riem'):
    x, y, cov, info = load_eeg_data(filename)
    x = change_window_size(x, window_size)

    if not diff:
        x = derivative2origianl(x)
    if clip:
        x, y = clip_onset_motion(x, filename, before_sec=1, after_sec=1)
    if freq is not None:
        x = freq_filter(x, freq)
    if method == 'riem':
        regularizer = pyriemann.estimation.Shrinkage()
        cov = sequence2covariance(x)
        cov = regularizer.transform(cov)
        x, x_mean = covariance2tangentspace(cov)
    elif method == 'cov':
        cov = sequence2covariance(x)
        x = convariance2vector(cov)
    elif method == 'raw':
        x = x.reshape((x.shape[0], -1))
    return x, y


def get_data_partial(filename, window_size: int = 320, freq=[5, 15]):
    if filename in ['data001.npz', 'data002.npz']:
        return partial(get_data, filename=filename, window_size=window_size, freq=freq, clip=False)
    elif filename in ['data_opti_101.npz', 'data_opti_102.npz']:
        return partial(get_data, filename=filename, window_size=window_size, freq=freq, clip=True)


def dataloader(train_file: str, valid_file: str, window_size: int = 320, freq=[5, 15]):
    get_train_data = get_data_partial(train_file, window_size, freq)
    get_valid_data = get_data_partial(valid_file, window_size, freq)

    xt_diff_riem, yt_diff_riem = get_train_data(diff=True, method='riem')
    xt_orig_riem, yt_orig_riem = get_train_data(diff=False, method='riem')
    xt_diff_cov, yt_diff_cov = get_train_data(diff=True, method='cov')
    xt_orig_cov, yt_orig_cov = get_train_data(diff=False, method='cov')
    xt_diff_raw, yt_diff_raw = get_train_data(diff=True, method='raw')
    xt_orig_raw, yt_orig_raw = get_train_data(diff=False, method='raw')

    xv_diff_riem, yv_diff_riem = get_valid_data(diff=True, method='riem')
    xv_orig_riem, yv_orig_riem = get_valid_data(diff=False, method='riem')
    xv_diff_cov, yv_diff_cov = get_valid_data(diff=True, method='cov')
    xv_orig_cov, yv_orig_cov = get_valid_data(diff=False, method='cov')
    xv_diff_raw, yv_diff_raw = get_valid_data(diff=True, method='raw')
    xv_orig_raw, yv_orig_raw = get_valid_data(diff=False, method='raw')

    data = {"train": {"x": [xt_diff_riem, xt_orig_riem, xt_diff_cov, xt_orig_cov, xt_diff_raw, xt_orig_raw],
                      "y": [yt_diff_riem, yt_orig_riem, yt_diff_cov, yt_orig_cov, yt_diff_raw, yt_orig_raw]},
            "valid": {"x": [xv_diff_riem, xv_orig_riem, xv_diff_cov, xv_orig_cov, xv_diff_raw, xv_orig_raw],
                      "y": [yv_diff_riem, yv_orig_riem, yv_diff_cov, yv_orig_cov, yv_diff_raw, yv_orig_raw]}}

    return data


if __name__ == '__main__':
    import time
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    classifiers = [SVC(C=0.1, gamma=0.5), LogisticRegression(max_iter=1000),
                   MLPClassifier(hidden_layer_sizes=(100, 100, 100)), RandomForestClassifier()]

    print("Loading datasets")
    data = dataloader("data001.npz", "data002.npz")
    print("Finished loading datasets")

    for cls in classifiers:
        print("Evaluating classifier: ", cls)
        for i, x0, y0, x1, y1 in enumerate(zip(data["train"]["x"], data["train"]["y"],
                                            data["valid"]["x"], data["valid"]["y"])):
            print("Training " + str(i+1) + " case")
            t0 = time.time()
            cls.fit(x0, y0)
            t1 = time.time()
            print("Evaluating " + str(i + 1) + " case")
            pred = cls.predict(x1)
            t2 = time.time()
            print(i + 1, " acc: ", accuracy_score(y1, pred), ' training time: ', t1 - t0, ' pred time: ', t2 - t1)