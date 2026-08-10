import numpy as np
from typing import Callable, Optional


def euclidean_dist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # (a-b)^2 = a^2 - 2ab + b^2
    x1_sum_sq = np.sum(x1 ** 2, axis=1, keepdims=True)
    x2_sum_sq = np.sum(x2 ** 2, axis=1, keepdims=True)

    distances_sq = x1_sum_sq - 2 * x1 @ x2.T + x2_sum_sq.T

    distances_sq = np.maximum(distances_sq, 0)
    return np.sqrt(distances_sq)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    u_abs = np.abs(u)
    return np.where(u_abs <= 1, (3 / 4) * (1 - u_abs ** 2), 0.0)


class WeightedKNearestNeighbors:
    def __init__(
        self,
        num_neighbors: int = 20,
        calc_distances: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_dist
    ):
        if num_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive number")
        self.num_neighbors = num_neighbors
        self.calc_distances_func = calc_distances
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.unique_classes: Optional[np.ndarray] = None
        self.epsilon = 1e-9

        self.last_x_test_for_animation: Optional[np.ndarray] = None
        self.last_knn_indexes_for_animation: Optional[np.ndarray] = None
        self.last_knn_distances_for_animation: Optional[np.ndarray] = None
        self.last_eff_distance_values_for_animation: Optional[np.ndarray] = None
        self.last_weights_for_animation: Optional[np.ndarray] = None
        self.last_predictions_for_animation: Optional[np.ndarray] = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
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
                self.last_eff_distance_values_for_animation = np.array([])
                self.last_weights_for_animation = np.array([])
                self.last_predictions_for_animation = np.array([])
            return np.ndarray([])

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        if self.x_train.shape[1] != x_test.shape[1]:
            raise ValueError("Test data shape doesn't match train data shape")

        distances_matrix = self.calc_distances_func(x_test, self.x_train)

        num_eff_neighbours = min(self.num_neighbors, self.x_train.shape[0])

        knn_indexes = np.argsort(distances_matrix, axis=1)[:, :num_eff_neighbours]
        knn_distances = np.take_along_axis(distances_matrix, knn_indexes, axis=1)
        knn_names = self.y_train[knn_indexes]

        eff_distance_values = knn_distances[:, num_eff_neighbours - 1]

        weights = np.zeros_like(knn_distances, dtype=float)

        mask_eff_distance_positive = eff_distance_values > self.epsilon
        mask_eff_distance_zero = ~mask_eff_distance_positive

        if np.any(mask_eff_distance_positive):
            eff_distance_expanded = eff_distance_values[
                mask_eff_distance_positive, np.newaxis]

            knn_distances_for_eff_distance_pos = knn_distances[
                mask_eff_distance_positive, :]

            u_values = (
                knn_distances_for_eff_distance_pos / eff_distance_expanded
            )
            weights[mask_eff_distance_positive, :] = epanechnikov_kernel(
                u_values
            )

        if np.any(mask_eff_distance_zero):
            knn_distances_for_eff_distance_zero = knn_distances[
                mask_eff_distance_zero, :]

            weights[mask_eff_distance_zero, :] = np.where(
                np.abs(knn_distances_for_eff_distance_zero) < self.epsilon,
                3/4,
                0.0
            )

        num_test_samples = x_test.shape[0]
        num_classes = len(self.unique_classes)
        weighted_votes = np.zeros(
            (num_test_samples, num_classes), dtype=float
        )

        for class_id, class_name in enumerate(self.unique_classes):
            weighted_votes[:, class_id] = np.sum(
                weights * (knn_names == class_name), axis=1
            )

        predicted_indexes_in_unique = np.argmax(weighted_votes, axis=1)
        predictions = self.unique_classes[predicted_indexes_in_unique]

        if store_for_animation:
            self.last_x_test_for_animation = x_test.copy()
            self.last_knn_indexes_for_animation = knn_indexes
            self.last_knn_distances_for_animation = knn_distances
            self.last_eff_distance_values_for_animation = eff_distance_values
            self.last_weights_for_animation = weights
            self.last_predictions_for_animation = predictions

        return predictions
