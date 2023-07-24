import numpy as np
from scipy.signal import stft, istft


def fivabss(x, nfft=512, maxiter=1000, tol=1e-6, nsou=None):
    nmic, nn = x.shape

    if nsou is None:
        nsou = nmic

    win = 2 * np.hanning(nfft) / nfft
    nol = int(3 * nfft / 4)

    X = np.zeros((nmic, nn, nfft // 2 + 1), dtype=np.complex128)

    for l in range(nmic):
        print(stft(x[l, :], nperseg=nfft+1, noverlap=nfft, window='hann')[2].shape)

        _, _, X[l, :, :] = stft(x[l, :], nperseg=nfft, noverlap=nol, window='hann')

    N = X.shape[1]
    nfreq = X.shape[2]
    epsi = 1e-6
    pObj = np.inf

    # Memory allocations
    Wp = np.zeros((nsou, nsou, nfreq), dtype=np.complex128)
    Q = np.zeros((nsou, nmic, nfreq), dtype=np.complex128)
    Xp = np.zeros((nsou, N, nfreq), dtype=np.complex128)
    S = np.zeros((nsou, N, nfreq), dtype=np.complex128)
    S2 = np.zeros((nsou, N, nfreq), dtype=np.complex128)
    Ssq = np.zeros((nsou, N))
    Ssq1 = np.zeros((nsou, N))
    Ssq3 = np.zeros((nsou, N))

    # Execute PCA and initialize
    for k in range(nfreq):
        Xmean = np.mean(X[:, :, k], axis=1)[:, np.newaxis]
        Rxx = np.dot((X[:, :, k] - Xmean), (X[:, :, k] - Xmean).conj().T) / N
        E, D = np.linalg.eig(Rxx)
        d = np.real(E)
        order = np.argsort(-d)[:nsou]
        E = E[:, order]
        D = np.diag(np.real(d[order] ** -0.5))
        Q[:, :, k] = np.dot(D, E.conj().T)
        Xp[:, :, k] = np.dot(Q[:, :, k], (X[:, :, k] - Xmean))

        Wp[:, :, k] = np.eye(nsou)

    # Start iterative learning algorithm
    for iter in range(maxiter):
        # Calculate outputs
        for k in range(nfreq):
            S[:, :, k] = np.dot(Wp[:, :, k], Xp[:, :, k])

        S2 = np.abs(S) ** 2
        Ssq = np.sum(S2, axis=2) ** 0.5
        Ssq1 = (Ssq + epsi) ** -1
        Ssq3 = Ssq1 ** 3

        for k in range(nfreq):
            # Calculate Hessian and nonlinear function
            Zta = np.diag(np.mean((Ssq1 - Ssq3 * S2[:, :, k]), axis=1))

            Phi = Ssq1 * S[:, :, k]

            # Update unmixing matrices
            Wp[:, :, k] = np.dot(Zta, Wp[:, :, k]) - np.dot(Phi, Xp[:, :, k].conj().T) / N

            # Decorrelation
            Wp[:, :, k] = np.linalg.solve(np.dot(Wp[:, :, k], Wp[:, :, k].conj().T), Wp[:, :, k])

        Obj = np.sum(Ssq) / (N * nsou * nfreq)
        dObj = (pObj - Obj) / np.abs(Obj)
        pObj = Obj

        if iter % 10 == 0:
            print(f'{iter} iterations: Objective={Obj}, dObj={dObj}')

        if abs(dObj) < tol:
            break

    # Correct scaling of unmixing filter coefficients
    W = np.zeros((nsou, nmic, nfreq), dtype=np.complex128)
    for k in range(nfreq):
        W[:, :, k] = np.dot(Wp[:, :, k], Q[:, :, k])
        W[:, :, k] = np.dot(np.diag(np.diag(np.linalg.pinv(W[:, :, k]))), W[:, :, k])

    # Calculate outputs
    S = np.zeros((nsou, N, nfreq), dtype=np.complex128)
    for k in range(nfreq):
        S[:, :, k] = np.dot(W[:, :, k], X[:, :, k])

    # Re-synthesize the obtained source signals
    y = np.zeros((nsou, nn))
    for k in range(nsou):
        _, y[k, :] = istft(S[k, :, :].T, nperseg=nfft, noverlap=nol, window=win)

    return y, W

# Example usage:
# y, W = fivabss(x)
