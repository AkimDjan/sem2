import numpy as np

class ShapeMismatchError(Exception):
    pass

def get_projections_components(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    
    if matrix.shape[0]!=matrix.shape[1]:
        raise ShapeMismatchError("matrix must be square")
    
    if matrix.shape[1]!=vector.shape[0]:
        raise ShapeMismatchError("size of matrix's column must be equal vector's length")

    if np.linalg.matrix_rank(matrix)!=matrix.shape[0]:
        return None, None

    abs_base_vectors = np.sqrt(np.add.reduce(matrix ** 2, axis=1))
    ort_proj = ((matrix @ vector) / (abs_base_vectors ** 2) * matrix.T).T
    ort_comp = vector[np.newaxis, ...] - ort_proj

    return ort_proj, ort_comp

#################################

matrix = np.array([[1, 2], [2, 4]])
vector = np.array([0, 1])

projections, orthogonals = get_projections_components(matrix, vector)

assert projections is None
assert orthogonals is None

#####

matrix = np.diag([2, 3])
vector = np.arange(start=1, stop=3)
projections_expected = np.array([[1, 0], [0, 2]])
orthogonals_expected = np.array([[0, 2], [1, 0]])
projections, orthogonals = get_projections_components(matrix, vector)


assert np.allclose(projections, projections_expected)
assert np.allclose(orthogonals, orthogonals_expected)

matrix = np.array([[1, 0], [1, 1]])
vector = np.array([0, 1])
projections_expected = np.array([[0, 0], [0.5, 0.5]])
orthogonals_expected = np.array([[0, 1], [-0.5, 0.5]])

projections, orthogonals = get_projections_components(matrix, vector)

assert np.allclose(projections, projections_expected)
assert np.allclose(orthogonals, orthogonals_expected)




