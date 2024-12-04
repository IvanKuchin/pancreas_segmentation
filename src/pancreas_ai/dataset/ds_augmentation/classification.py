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

    def __pad_to_min_shape(self, arr:npt.NDArray, min_shape:npt.NDArray) -> npt.NDArray:
        assert min_shape.shape == (3,) and min_shape.dtype == np.int32 and arr.ndim == 3

        pad_arr = np.ceil((min_shape - arr.shape) / 2.0).astype(dtype=np.int32)
        pad_arr = np.where(pad_arr < 0, 0, pad_arr)

        pad = []
        for pad_item in pad_arr:
            pad.append([pad_item, pad_item])

        result = np.pad(arr, pad, mode="constant")

        assert np.any(result.shape < min_shape) == False, f"padding failed to reach min shape: result.shape:{result.shape} < min_shape:{min_shape}"

        return result


    def random_resize(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Output size will vary based pn config parameters
        [width +- randpm, height +- random, depth +- random]
        If predefined size is required, cropping or padding is needed
        """
        image_size = np.array([self.config.IMAGE_DIMENSION_X, self.config.IMAGE_DIMENSION_Y, self.config.IMAGE_DIMENSION_Z])
        rand_arr = (np.random.rand(3) - 0.5) * self.config.AUGMENTATION_RANDOM_SCALE
        new_size = (image_size * (1 + rand_arr)).astype(dtype = np.int32)
        resized_data, resized_label = resize_3d.resize_3d_image(data, new_size), mask
        padded_data = self.__pad_to_min_shape(resized_data, image_size)

        return padded_data, resized_label
    