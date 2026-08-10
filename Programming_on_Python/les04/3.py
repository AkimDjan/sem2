import numpy as np

def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:

    if threshold < 1:
        raise ValueError("Threshold must be more or equal than 1")

    all_image = np.unique_values(image)
    freq = np.zeros_like(all_image)

    for i in range(freq.size):
        freq[i] = np.sum(np.abs(all_image - all_image[i]) <= threshold)

    most_frequent = np.uint8(all_image[np.argmax(freq)])
    percent = float(np.max(freq) / image.size)

    return most_frequent, percent*100