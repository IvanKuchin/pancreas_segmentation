import numpy as np

def random_flip_data_and_label(data, label) -> tuple[np.ndarray, np.ndarray]:
    for i in range(len(data.shape)):
        if np.random.rand() > 0.5:
            data = np.flip(data, i)
            label = np.flip(label, i)
    return data, label

def random_flip_data(data) -> np.ndarray:
    for i in range(len(data.shape)):
        if np.random.rand() > 0.5:
            data = np.flip(data, i)
    return data
