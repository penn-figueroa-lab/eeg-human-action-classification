import numpy as np


class ICA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.x_mean = None
        self.x_wht = None
        self.Wht = None
        self.Wht_inv = None
        self.W = None
        self.W_inv = None
        self.exclude = []

    def whitening(self, x):
        n, m = x.shape
        self.x_mean = np.mean(x, axis=1, keepdims=True)
        x_cntr = x - self.x_mean
        x_cov = (x_cntr @ x_cntr.T) / (m - 1)  # sample covariance
        d, E = np.linalg.eig(x_cov)
        self.Wht = E @ np.diag(1 / np.sqrt(d)) @ E.T
        self.Wht_inv = E @ np.diag(np.sqrt(d)) @ E.T
        self.x_wht = self.Wht @ x_cntr

    def fit(self, x, thresh=1e-8, iterations=5000):
        n, m = x.shape
        self.whitening(x)
        self.W = np.ones((self.n_components, n)) / np.sqrt(n)

        for c in range(self.n_components):
            i = 0
            lim = 100
            while (lim > thresh) & (i < iterations):
                wx = np.tanh(self.W[c, :].reshape(1, n) @ self.x_wht)
                wn = (self.x_wht * wx).mean(axis=1) - (1 - wx ** 2).mean() * self.W[c, :]
                wn = wn - wn @ self.W[:c].T @ self.W[:c]
                wn = wn / np.linalg.norm(wn, keepdims=True)
                lim = np.abs(np.abs((wn * self.W[c, :]).sum()) - 1)
                self.W[c, :] = wn
                i += 1

        self.W_inv = np.linalg.pinv(self.W)
        return self.W @ self.x_wht

    def apply(self, x=None):
        if x is not None:
            self.whitening(x)
        W_inv = self.W_inv.copy()
        W_inv[:, self.exclude] = 0
        return (self.Wht_inv @ (W_inv @ (self.W @ self.x_wht))) + self.x_mean
