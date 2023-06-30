import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    print(mean)
    centered =  x - mean
    return centered, mean

def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    print(mean)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n


def whiten(x):
    # Calculate the covariance matrix
    coVarM = covariance(X)

    # Single value decoposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, X)

    return Xw, whiteM
