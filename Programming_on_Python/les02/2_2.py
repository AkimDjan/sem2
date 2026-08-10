def transpose(matrix: list[list[float]]) -> list[list[float]]:
    new_matrix=[[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(new_matrix)):
        for j in range(len(new_matrix[0])):
            new_matrix[i][j]=matrix[j][i]
    return new_matrix

matrix = [
    [-2, -5],
    [-5, -6],
    [-5, 3],
]
reference = [
    [-2, -5, -5],
    [-5, -6, 3],
]

transposed = transpose(matrix)

assert transposed is not matrix
assert all(
    all(
        num_res == num_ref 
        for num_res, num_ref in zip(row_res, row_ref)
    )
    for row_res, row_ref in zip(transposed, reference)
)