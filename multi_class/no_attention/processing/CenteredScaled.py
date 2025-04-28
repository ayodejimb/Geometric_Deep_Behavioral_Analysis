
import numpy as np
from scipy.linalg import svd

def CenteredScaled(X):
    X = X- np.mean(X)
   # the "centered" Frobenius norm
    normX = np.linalg.norm(X,'fro') 
    # scale to equal (unit) norm
    if (normX !=0):
        X = X / normX

    return X

def Normalize(X):
    normX = np.linalg.norm(X, 'fro') 
    # scale to equal (unit) norm
    if (normX !=0):
        X = X / normX

    return X

def ref_rot(X,ref):
    n_X = np.transpose(X,[1,0])
    n_X = CenteredScaled(n_X)
    ref = CenteredScaled(ref)
    H = np.matmul(n_X,ref)
    u,s,v = svd(H,full_matrices=False)
    rot = v.T*u.T
    new_X = np.matmul(X,rot)
    return new_X