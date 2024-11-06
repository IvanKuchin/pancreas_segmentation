import numpy as np
import cv2

def __rotate_CV(image, angel , interpolation):
    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #interpolation cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angel,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=interpolation)
    return rotated

def __rotate_data(angle: float, axis: int, data: np.ndarray) -> np.ndarray:
    unstacked = np.split(data, data.shape[axis], axis)
    unstacked = [np.squeeze(x) for x in unstacked]
    rotated = [__rotate_CV(x, angle, cv2.INTER_LINEAR) for x in unstacked]
    return np.stack(rotated, axis)

def random_rotate_data_and_label(data: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for i in range(len(data.shape)):
        if np.random.rand() > 0.5:
            data = np.flip(data, i)
            label = np.flip(label, i)
    return data, label

def random_rotate_data(angle1: float, angle2: float, data: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(angle1, angle2)
    axis = np.random.randint(0, data.ndim)

    return __rotate_data(angle, axis, data)
