import numpy as np


def accuracy(
    true_targets: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if true_targets.shape != prediction.shape:
        raise ValueError("True_targets and predictions must be the same size")
    if true_targets.ndim != 1:
        raise ValueError("Arrays true_targets and predictions must be one-dimensional")
    if true_targets.size == 0:
        return 1.0

    correct_predictions = np.sum(true_targets == prediction)

    return correct_predictions / true_targets.size
