import numpy as np
from typing import Callable, Any


def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], float],
) -> np.ndarray:
    try:
        values = np.array([key(x) for x in data])
        if values.ndim != 1:
            raise ValueError("Key must return scalar values")
    except Exception as e:
        raise TypeError(
            f"Error occurred while applying the function: {e}"
        )

    if values.size == 0:
        return np.array([], dtype=int)

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    low_bound = q1 - 1.5 * iqr
    up_bound = q3 + 1.5 * iqr

    outlier_mask = (values < low_bound) | (values > up_bound)
    outlier_indexes = np.nonzero(outlier_mask)[0]

    return outlier_indexes
