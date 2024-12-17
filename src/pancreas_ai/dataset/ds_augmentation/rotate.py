import numpy as np
import scipy.ndimage as ndimage

def __rotate_scipy(data: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(data, angle, reshape=False, order=1)

def rotate_data(angle: float, axis: int, data: np.ndarray) -> np.ndarray:
    assert data.ndim == 3

    unstacked = np.split(data, data.shape[axis], axis)
    unstacked = [np.squeeze(x) for x in unstacked]
    rotated = [__rotate_scipy(x, angle) for x in unstacked]
    return np.stack(rotated, axis)

def random_rotate_data_and_label(angle1: float, angle2: float, data: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.uniform(angle1, angle2)
    axis = np.random.randint(0, data.ndim)

    return rotate_data(angle, axis, data), rotate_data(angle, axis, label)

def random_rotate_data(data: np.ndarray, angle1: float, angle2: float) -> np.ndarray:
    angle = np.random.uniform(angle1, angle2)
    axis = np.random.randint(0, data.ndim)

    return rotate_data(angle, axis, data)
