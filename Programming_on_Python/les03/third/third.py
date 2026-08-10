import numpy as np

def get_extremum_indices(
    ordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(ordinates) <3: 
        raise ValueError("massive's len must be longer than 2")
    index_min = []
    index_max = []
    for i in range(1, len(ordinates)-1): #тк краевые точки не участвуют в анализе
        if (ordinates[i-1] > ordinates[i]) and (ordinates [i] < ordinates[i+1]):
            index_min += [i]
        if (ordinates[i-1] < ordinates[i]) and (ordinates [i] > ordinates[i+1]):
            index_max += [i]
    return np.array(index_min) , np.array(index_max)

#########################################################

ordinates = np.sin(2 * np.linspace(0, 4 * np.pi, 1000))
indices_min_expected = np.array([187, 437, 687, 937], dtype=np.int32)
indices_max_expected = np.array([ 62, 312, 562, 812], dtype=np.int32)

indices_min, indices_max = get_extremum_indices(ordinates)

assert np.allclose(indices_min, indices_min_expected)
assert np.allclose(indices_max, indices_max_expected)

ordinates = np.array([1, 3, 2, 4, 1, 5, 0])
minima, maxima = get_extremum_indices(ordinates)
print(minima,maxima)