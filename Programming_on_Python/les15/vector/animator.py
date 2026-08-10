import numpy as np
from typing import Union, Optional, Tuple, List

from my_module.algorithms.knn import KNearestNeighbors
from my_module.algorithms.weighted_knn import WeightedKNearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


plt.style.use("ggplot")


class GifSaveError(Exception):
    pass


class AnimationKNN:
    def __init__(self):
        self.figure = None
        self.axis = None
        self.knn_model: Optional[
            Union[KNearestNeighbors, WeightedKNearestNeighbors]
        ] = None

        self.animation_frame_data: List[dict] = []
        self.class_colors = ['blue', 'red', 'green', 'brown', 'purple', 'black']

    def _get_class_color(self, label) -> str:
        try:
            if self.knn_model and self.knn_model.unique_classes:
                classes = list(self.knn_model.unique_classes)
            else:
                classes = []

            if label in classes:
                index = classes.index(label)
            else:
                index = int(label)
        except Exception:
            index = hash(str(label))

        return self.class_colors[index % len(self.class_colors)]

    def _prepare_animation_data(
            self,
            knn_model: Union[KNearestNeighbors, WeightedKNearestNeighbors],
            x_animation_set: np.ndarray,
            y_animation_true: np.ndarray
    ):
        self.animation_frame_data = []
        self.knn_model = knn_model

        _ = knn_model.predict(x_animation_set, store_for_animation=True)

        if (knn_model.last_x_test_for_animation is None or
                knn_model.last_knn_indexes_for_animation is None or
                knn_model.last_predictions_for_animation is None):
            raise RuntimeError("Data for animation wasn't saved")

        num_frames = knn_model.last_x_test_for_animation.shape[0]

        for i in range(num_frames):
            frame_info = {
                "test_point_coords": knn_model.last_x_test_for_animation[i],
                "true_label": y_animation_true[i],
                "predicted_label": knn_model.last_predictions_for_animation[i],
            }

            neighbor_indexes_for_point = (
                knn_model.last_knn_indexes_for_animation[i]
            )
            frame_info["neighbor_coords"] = (
                knn_model.x_train[neighbor_indexes_for_point]
            )
            frame_info["neighbor_labels"] = (
                knn_model.y_train[neighbor_indexes_for_point]
            )

            if isinstance(knn_model, WeightedKNearestNeighbors):
                if knn_model.last_eff_distance_values_for_animation is not None:
                    frame_info["eff_distance_value"] = (
                        knn_model.last_eff_distance_values_for_animation[i]
                    )

            elif isinstance(knn_model, KNearestNeighbors):
                if knn_model.last_knn_distances_for_animation is not None:
                    frame_info["eff_distance_value"] = (
                        knn_model.last_knn_distances_for_animation[i, -1]
                    )

            self.animation_frame_data.append(frame_info)

    def _update_frame(self, frame_id: int, num_all_frames: int) -> Tuple:
        if not self.animation_frame_data or frame_id >= len(
                self.animation_frame_data):
            return []

        self.axis.cla()
        current_frame_data = self.animation_frame_data[frame_id]

        if (self.knn_model is not None and
                self.knn_model.x_train is not None and
                self.knn_model.y_train is not None):
            if self.knn_model.unique_classes is not None:
                unique_plot_labels = self.knn_model.unique_classes
            else:
                unique_plot_labels = np.unique(self.knn_model.y_train)

            for label_val in unique_plot_labels:
                mask = (self.knn_model.y_train == label_val)
                self.axis.scatter(
                    self.knn_model.x_train[mask, 0],
                    self.knn_model.x_train[mask, 1],
                    color=self._get_class_color(label_val),
                    alpha=0.3,
                    s=30,
                )

        test_coords = current_frame_data["test_point_coords"]
        pred_label = current_frame_data["predicted_label"]
        true_label = current_frame_data["true_label"]

        self.axis.scatter(
            test_coords[0],
            test_coords[1],
            color=self._get_class_color(pred_label if pred_label else 0),
            s=80,
            edgecolor='black',
            marker='*',
        )

        neighbor_coords = current_frame_data["neighbor_coords"]
        neighbor_labels = current_frame_data["neighbor_labels"]
        for i, coord in enumerate(neighbor_coords):
            self.axis.scatter(
                coord[0],
                coord[1],
                color=self._get_class_color(neighbor_labels[i]),
                s=50,
                edgecolor='black',
                alpha=0.6,
                marker='o'
            )

            self.axis.plot(
                [test_coords[0], coord[0]],
                [test_coords[1], coord[1]],
                color='darkred',
                linestyle='--',
                linewidth=0.7,
                alpha=0.6
            )

        radius = 0.0
        if "eff_distance_value" in current_frame_data:
            radius = current_frame_data["eff_distance_value"]

        if radius > 1e-9:
            circle = Circle(
                (test_coords[0], test_coords[1]),
                radius,
                facecolor='lightgreen',
                alpha=0.3,
                edgecolor='darkgray',
            )
            self.axis.add_patch(circle)

        status = "Correct" if pred_label == true_label else "Incorrect"
        self.axis.set_title(
            f"Dot {frame_id + 1}/{num_all_frames}. "
            f"True: {true_label}, Predicted: {pred_label} ({status})"
        )
        self.axis.set_xlabel("Axis x")
        self.axis.set_ylabel("Axis y")

        all_artists = (
            self.axis.collections
            + self.axis.lines
            + self.axis.patches
            + [self.axis.title]
            + self.axis.texts
        )
        return tuple(all_artists)

    def create_animation(
            self,
            knn_model: Union[KNearestNeighbors, WeightedKNearestNeighbors],
            x_animation_set: np.ndarray,
            y_animation_true: np.ndarray,
            path_to_save: str = "",
    ) -> FuncAnimation:
        if x_animation_set.shape[0] == 0:
            figure = plt.figure()
            plt.close(figure)
            return FuncAnimation(figure, lambda x: [], frames=0)

        self._prepare_animation_data(knn_model, x_animation_set, y_animation_true)

        if not self.animation_frame_data:
            figure = plt.figure()
            plt.close(figure)
            return FuncAnimation(figure, lambda x: [], frames=0)

        self.figure, self.axis = plt.subplots(figsize=(10, 8))
        num_frames = len(self.animation_frame_data)

        animation = FuncAnimation(
            self.figure,
            self._update_frame,
            frames=num_frames,
            fargs=(num_frames,),
            interval=1000,
            blit=False,
            repeat=True
        )

        if path_to_save:
            try:
                fps_value = 1
                animation.save(path_to_save, writer='pillow', fps=fps_value)
            except:
                raise GifSaveError('Error occurred while saving the animation')

        return animation
