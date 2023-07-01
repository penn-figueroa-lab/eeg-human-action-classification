import numpy as np


class CCA:
    def __init__(self):
        self.x_mean = None
        self.x_cntr = None
        self.y_mean = None
        self.y_cntr = None
        self.W = None
        self.W_inv = None
        self.exclude = []

    def centering(self, x, y):
        self.x_mean = np.mean(x, axis=1, keepdims=True)
        self.x_cntr = x - self.x_mean
        self.y_mean = np.mean(y, axis=1, keepdims=True)
        self.y_cntr = y - self.y_mean

    def fit(self, signal):
        n, m = signal.shape
        self.centering(signal[:, 1:], signal[:, :m - 1])
        xy = np.concatenate([self.x_cntr, self.y_cntr],axis=0)
        C = (xy @ xy.T) / (m - 1)
        Cxx = C[:n, :n]
        Cxy = C[:n, n:]
        Cyy = C[n:, n:]
        Cyx = C[n:, :n]
        CCx = np.linalg.inv(Cxx) @ Cxy @ np.linalg.inv(Cyy) @ Cyx
        rho_squared, wx = np.linalg.eig(CCx)
        self.W = wx.T
        self.W_inv = np.linalg.pinv(self.W)

        return self.W @ self.x_cntr

    def apply(self, x=None):
        if x is not None:
            n, m = x.shape
            self.centering(x[:, 1:], x[:, :m - 1])
        W_inv = self.W_inv.copy()
        W_inv[:, self.exclude] = 0
        return (W_inv @ (self.W @ self.x_cntr)) + self.x_mean
