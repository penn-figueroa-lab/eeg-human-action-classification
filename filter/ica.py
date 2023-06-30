import numpy
import numpy as np


class ICA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.x_mean = None
        self.x_wht = None
        self.Wht = None
        self.Wht_inv = None
        self.W = None

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
        self.W = np.random.rand(self.n_components, n)

        for c in range(self.n_components):
            w = self.W[c, :].copy().reshape(n, 1)
            w = w / np.linalg.norm(w, keepdims=True)

            i = 0
            lim = 100
            while (lim > thresh) & (i < iterations):
                # Dot product of weight and signal
                ws = np.dot(w.T, self.x_wht)

                # Pass w*s into contrast function g
                wg = np.tanh(ws).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws)))

                # Update weights
                wNew = (self.x_wht * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights
                wNew = wNew - np.dot(np.dot(wNew, self.W[:c].T), self.W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            self.W[c, :] = w.T

    def fastIca(self, x, alpha=1, thresh=1e-8, iterations=5000):
        n, m = x.shape

        # Initialize random weights
        W = np.random.rand(self.n_components, n)

        for c in range(self.n_components):
            w = W[c, :].copy().reshape(n, 1)
            w = w / np.linalg.norm(w, keepdims=True)

            i = 0
            lim = 100
            while (lim > thresh) & (i < iterations):
                # Dot product of weight and signal
                ws = np.dot(w.T, x)

                # Pass w*s into contrast function g
                wg = np.tanh(ws * alpha).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws))) * alpha

                # Update weights
                wNew = (x * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W[c, :] = w.T
        return W
