import numpy.typing as npt
import numpy as np

from . import interface
from . import flip
from . import crop

class Augment(interface.IAugment):
    def __init__(self):
        pass

    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return crop.random_crop_data(data, inside_width, inside_height, inside_depth), mask

    def random_flip(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return flip.random_flip_data(data), mask
