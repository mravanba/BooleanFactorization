import numpy as np
import pdb



def log_ratio(x):
    return np.log(x) - np.log(1. - x)

def get_random_matrices(M, N, K, p_x_1 = .5, p_y_1 = .5, p_flip = 0, p_observe = .1):
    X = (np.random.rand(M, K) < p_x_1).astype(int)
    Y = (np.random.rand(N, K) < p_y_1).astype(int)
    Z = (X.dot(Y.T) > 0).astype(int)
    mask = np.random.rand(M,N) < p_observe
    O = Z.copy()
    
    flip = np.random.rand(M,N) < p_flip
    O[flip] = 1 - O[flip]
    mats = {'X':X, 'Y':Y, 'Z':Z, 'O':O, 'mask':mask}
    return mats


def hamming(X,Y):
    return np.sum(np.abs(X - Y) > 1e-5)

def density(X):
    return np.sum(X)/float(np.prod(X.shape))


def rec_error(Z, Zh):
    return np.sum(np.abs(Z - Zh))/float(np.prod(Z.shape))


def read_csv(fname, delimiter = ','):
    mat = np.genfromtxt(fname, delimiter=delimiter)
    return mat
