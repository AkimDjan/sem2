import numpy as np

class ShapeMismatchError(Exception):
    pass

def sum_arrays_naive(
    lhs: list[float],
    rhs: list[float],
) -> list[float]:
    if len(lhs) != len(rhs):
        raise ShapeMismatchError
    
    return [
        elem_lhs + elem_rhs for elem_lhs, elem_rhs in zip(lhs, rhs)
    ]

#================================== 1 задача 
def sum_arrays_vectorized(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    if len(lhs) != len(rhs):
        raise ShapeMismatchError
    res=lhs+rhs
    return res
#==================================



def compute_poly_naive(abscissa: list[float]) -> list[float]:
    return [3 * (x ** 2) + 2 * x + 1 for x in abscissa]

#================================== 2 задача
def compute_poly_vectorized(abscissa: np.ndarray) -> np.ndarray:
    return np.array(3*(abscissa**2)+2*abscissa+1)
#==================================


def get_mutual_l2_distances_naive(
    lhs: list[list[float]],
    rhs: list[list[float]],
) -> list[list[float]]:    
    if len(lhs[0]) != len(rhs[0]):
        raise ShapeMismatchError
    return [
        [
            sum(
                (lhs[i][k] - rhs[j][k]) ** 2 for k in range(len(lhs[0]))
            ) ** 0.5
            for j in range(len(rhs))
        ]
        for i in range(len(lhs))
    ]

#================================== 3 задача
def get_mutual_l2_distances_vectorized(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    if len(lhs[0]) != len(rhs[0]):
        raise ShapeMismatchError
    result=np.zeros(shape=(len(lhs),len(rhs)))
    for i in range(len(lhs)):
        for j in range(len(rhs)):
            result[i][j]=(np.add.reduce((lhs[i]-rhs[j])**2))**0.5
    return result
#==================================
