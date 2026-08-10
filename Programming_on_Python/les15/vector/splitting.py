import numpy as np
from typing import List, Tuple


def initial_shuffle(
        features: np.ndarray,
        targets: np.ndarray,
        shuffle_flag: bool
) -> Tuple[np.ndarray, np.ndarray]:
    current_features = features
    current_targets = targets
    if shuffle_flag:
        indexes = np.arange(features.shape[0])
        np.random.shuffle(indexes)
        current_features = features[indexes]
        current_targets = targets[indexes]
    return current_features, current_targets


def append_class_data_to_lists(
        class_features_for_class: np.ndarray,
        class_targets_for_class: np.ndarray,
        num_train_for_class: int,
        train_features_list: List[np.ndarray],
        train_targets_list: List[np.ndarray],
        test_features_list: List[np.ndarray],
        test_targets_list: List[np.ndarray]
) -> None:
    current_train_features = class_features_for_class[:num_train_for_class]
    current_train_targets = class_targets_for_class[:num_train_for_class]
    current_test_features = class_features_for_class[num_train_for_class:]
    current_test_targets = class_targets_for_class[num_train_for_class:]

    if len(current_train_targets) > 0:
        train_features_list.append(current_train_features)
        train_targets_list.append(current_train_targets)
    if len(current_test_targets) > 0:
        test_features_list.append(current_test_features)
        test_targets_list.append(current_test_targets)


def concatenate_final_sets(
        features_list: List[np.ndarray],
        targets_list: List[np.ndarray],
        original_features: np.ndarray,
        original_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if features_list and any(arr.size > 0 for arr in features_list):
        final_features = np.concatenate([arr for arr in features_list if arr.size > 0], axis=0)
    else:
        final_features = np.ndarray([])

    if targets_list and any(arr.size > 0 for arr in targets_list):
        final_targets = np.concatenate([arr for arr in targets_list if arr.size > 0], axis=0)
    else:
        final_targets = np.ndarray([])
    return final_features, final_targets


def final_shuffle(
        features_set: np.ndarray,
        targets_set: np.ndarray,
        shuffle_flag: bool
) -> Tuple[np.ndarray, np.ndarray]:
    current_features = features_set
    current_targets = targets_set
    if shuffle_flag:
        permutation = np.random.permutation(len(current_targets))
        current_features = current_features[permutation]
        current_targets = current_targets[permutation]
    return current_features, current_targets


def train_test_split(
        features: np.ndarray,
        targets: np.ndarray,
        train_ratio: float = 0.8,
        shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1).")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "Number of samples in features does not match number of samples in targets"
        )

    shuffled_features, shuffled_targets = initial_shuffle(features, targets, shuffle)

    train_features_list = []
    test_features_list = []
    train_targets_list = []
    test_targets_list = []

    unique_classes = np.unique(shuffled_targets)

    for class_name in unique_classes:
        class_mask = (shuffled_targets == class_name)
        class_features_for_cls = shuffled_features[class_mask]
        class_targets_for_cls = shuffled_targets[class_mask]

        num_cls_samples = len(class_targets_for_cls)
        if num_cls_samples == 0:
            continue

        num_train_for_cls = int(round(num_cls_samples * train_ratio))

        append_class_data_to_lists(class_features_for_cls, class_targets_for_cls, num_train_for_cls,
                                   train_features_list, train_targets_list, test_features_list, test_targets_list)

    final_train_features, final_train_targets = concatenate_final_sets(train_features_list, train_targets_list,
                                                                       features, targets)

    final_test_features, final_test_targets = concatenate_final_sets(test_features_list, test_targets_list, features,
                                                                     targets)

    final_train_features, final_train_targets = final_shuffle(final_train_features, final_train_targets, shuffle)

    final_test_features, final_test_targets = final_shuffle(final_test_features, final_test_targets, shuffle)

    return final_train_features, final_train_targets, final_test_features, final_test_targets
