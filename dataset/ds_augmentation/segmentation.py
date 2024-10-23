import numpy.typing as npt
import numpy as np

import dataset.ds_augmentation.interface as interface
from dataset.ds_augmentation.crop import random_crop_data_and_label

class Augment(interface.IAugment):
    def __init__(self):
        pass

    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return random_crop_data_and_label(data, inside_width, inside_height, inside_depth)

