from random import randint

columns_amount = 1000           # число колонок в матрице для тестирования
rows_amount = 500               # число строк в матрице для тестирования

bottom = -10                    # нижняя граница значений чисел в матрице
top = 10                        # верхняя граница значений чисел в матрице

class ShapeMismatchError(Exception):
    """Возбуждается, если матрицы не могут быть перемножены."""


def multiply_matrices(
    lhs: list[list[float]],
    rhs: list[list[float]],
) -> list[list[float]]:
    if len(lhs[0])!=len(rhs): #left row size, right column size
        raise ShapeMismatchError("If you want to multiply matrixes, left matrix's row's size must be equal right matrix's column's size")
    result=[[0 for _ in range(len(lhs))] for _ in range((len(rhs[0])))]
    for i in range(len(lhs)):
        for j in range(len(rhs[0])):
            for k in range(len(rhs)):
                result[i][j]+=lhs[i][k]*rhs[k][j]
    return result
#########################

lhs = [
    [7, -1, -4],
    [-1, 5, -1],
]
rhs = [
    [-2, -5],
    [-5, -6],
    [-5, 3],
]
reference = [
    [11, -41],
    [-18, -28]
]

result = multiply_matrices(lhs, rhs)
assert all(
    all(
        num_res == num_ref 
        for num_res, num_ref in zip(row_res, row_ref)
    )
    for row_res, row_ref in zip(result, reference)
)

was_raised = False

try:
    result = multiply_matrices(lhs, rhs[:2])

except ShapeMismatchError:
    was_raised = True

assert was_raised