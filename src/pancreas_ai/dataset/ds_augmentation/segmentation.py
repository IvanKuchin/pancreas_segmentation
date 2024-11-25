import numpy.typing as npt
import numpy as np

from . import interface
from . import crop
from . import flip

class Augment(interface.IAugment):
    def __init__(self, config: dict):
        self.config = config

    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return crop.random_crop_data_and_label(data, inside_width, inside_height, inside_depth)

    def random_flip(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return flip.random_flip_data_and_label(data, mask)
