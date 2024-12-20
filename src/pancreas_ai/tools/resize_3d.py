import numpy as np
import numpy.typing as npt
from skimage.transform import resize

def __resize_along_axis_np(inp: npt.NDArray, new_size: npt.NDArray, axis_along: int) -> npt.NDArray:
    unstacked = np.split(inp, inp.shape[axis_along], axis_along)
    imgs = [np.squeeze(x) for x in unstacked]
    resized_imgs = [resize(_, new_size[:2], order=0, mode='edge') for _ in imgs]

    result = np.stack(resized_imgs, axis = axis_along)
    return result

def resize_3d_image(image: npt.NDArray, new_size: npt.NDArray) -> npt.NDArray:
    """
    Resize a 3D image to the specified dimensions
    """
    assert new_size.shape == (3,)
    assert new_size.dtype == np.int32 or new_size.dtype == np.int64
    assert image.ndim == 3
    zoomed_img = __resize_along_axis_np(image, new_size[:2], 2)
    zoomed_img = __resize_along_axis_np(zoomed_img, new_size[1:], 0)
    return zoomed_img
