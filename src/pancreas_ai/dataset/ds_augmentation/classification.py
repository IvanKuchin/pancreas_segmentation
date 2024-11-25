import numpy.typing as npt
import numpy as np

from . import interface
from . import flip
from . import crop
from . import rotate
from pancreas_ai.tools import resize_3d

class Augment(interface.IAugment):
    def __init__(self, config):
        self.config = config

    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return crop.random_crop_data(data, inside_width, inside_height, inside_depth), mask

    def random_flip(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return flip.random_flip_data(data), mask

    def random_rotate(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return rotate.random_rotate_data(data), mask

    def random_resize(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        new_size = np.array([self.config.IMAGE_DIMENSION_X, self.config.IMAGE_DIMENSION_Y, self.config.IMAGE_DIMENSION_Z])
        return resize_3d.resize_3d_image(data, new_size), mask
    