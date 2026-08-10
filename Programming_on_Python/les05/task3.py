import numpy as np

class ShapeMismatchError(Exception):
    pass

def adaptive_filter(
    Vs : np.ndarray, 
    Vj : np.ndarray, 
    diag_A : np.ndarray
) -> np.ndarray:
    
    if Vs.shape[0] != Vj.shape[0]:
        raise ShapeMismatchError("Sizes of matrixes must be equal")
    
    if Vj.shape[1] != diag_A.shape[0]:
        raise ShapeMismatchError("Sizes of matrixes must be equal")
    
    Vjh = (np.conj(Vj)).T
    A = np.diag(diag_A)
    b = np.linalg.inv(np.eye((Vjh @ Vj @ A).shape[0]) + Vjh @ Vj @ A)
    y =  Vs - (Vj @ (b @ (Vjh @ Vs)))

    return y

with open('source/diag_A_data.npy', 'rb') as f:
    diag_A = np.load(f)

with open('source/Vj_data.npy', 'rb') as f:
    Vj = np.load(f)

with open('source/Vs_data.npy', 'rb') as f:
    Vs = np.load(f)

with open('source/y_data.npy', 'rb') as f:
    y_check = np.load(f)

y = adaptive_filter(Vs, Vj, diag_A)
assert np.allclose(y, y_check)