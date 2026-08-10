import numpy as np
from typing import Callable, Optional


def euclidean_dist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # (a-b)^2 = a^2 - 2ab + b^2
    x1_sum_sq = np.sum(x1 ** 2, axis=1, keepdims=True)
    x2_sum_sq = np.sum(x2 ** 2, axis=1, keepdims=True)

    distances_sq = x1_sum_sq - 2 * x1 @ x2.T + x2_sum_sq.T

    distances_sq = np.maximum(distances_sq, 0)

    return np.sqrt(distances_sq)


class KNearestNeighbors:
    def __init__(
        self,
        num_neighbors: int = 5,
        calc_distances: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_dist
    ):
        if num_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive number")
        self.num_neighbors = num_neighbors
        self.calc_distances_func = calc_distances
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.unique_classes: Optional[np.ndarray] = None

        self.last_x_test_for_animation: Optional[np.ndarray] = None
        self.last_knn_indexes_for_animation: Optional[np.ndarray] = None
        self.last_knn_distances_for_animation: Optional[np.ndarray] = None
        self.last_predictions_for_animation: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Number of samples in x_train and y_train must be equal"
            )
        if x_train.shape[0] == 0:
            raise ValueError("Training set mustn't be empty")

        self.x_train = x_train
        self.y_train = y_train
        self.unique_classes = np.unique(y_train)

    def predict(
            self,
            x_test: np.ndarray,
            store_for_animation: bool = False
    ) -> np.ndarray:
        if (self.x_train is None
                or self.y_train is None
                or self.unique_classes is None):
            raise RuntimeError("Call fit function before")

        if x_test.shape[0] == 0:
            if store_for_animation:
                self.last_x_test_for_animation = np.array([])
                self.last_knn_indexes_for_animation = np.array([])
                self.last_knn_distances_for_animation = np.array([])
                self.last_predictions_for_animation = np.array([])
            return np.ndarray([])

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        if self.x_train.shape[1] != x_test.shape[1]:
            raise ValueError("Test data shape doesn't match train data shape")

        distances_matrix = self.calc_distances_func(x_test, self.x_train)
        num_eff_neighbours = min(self.num_neighbors, self.x_train.shape[0])
        knn_indexes = np.argsort(distances_matrix, axis=1)[:, :num_eff_neighbours]

        knn_names = self.y_train[knn_indexes]
        knn_distances = np.take_along_axis(
            distances_matrix, knn_indexes, axis=1
        )

        num_test_samples = x_test.shape[0]
        num_classes = len(self.unique_classes)

        class_votes = np.zeros((num_test_samples, num_classes), dtype=int)

        for class_id, class_value in enumerate(self.unique_classes):
            class_votes[:, class_id] = np.sum(knn_names == class_value, axis=1)

        predicted_indexes = np.argmax(class_votes, axis=1)
        predictions = self.unique_classes[predicted_indexes]

        if store_for_animation:
            self.last_x_test_for_animation = x_test
            self.last_knn_indexes_for_animation = knn_indexes
            self.last_knn_distances_for_animation = knn_distances
            self.last_predictions_for_animation = predictions

        return predictions
