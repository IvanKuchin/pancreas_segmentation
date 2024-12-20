import os
import sys
import time

import numpy as np
import numpy.typing as npt
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tools import resize_3d
from pancreas_ai import config


DEBUG_DATALOADER = True
DEBUG_DATA_LOADING_PERFORMANCE = False

def __cutout_and_resize_tensor(tensor: npt.NDArray[np.float32], top_left: npt.NDArray[np.float32] , bottom_right: npt.NDArray[np.float32]) -> npt.NDArray:
    """
    cutout and resize 3-d tensor with shape [w,h,d]
    1) cut overheads off from top_left to bottom_right + 1
       top_left and bottom_right will be present in the final shape
    2) resize shape from step(1) to final shape
       final shape taken form the config
    """
    assert tensor.ndim == 3
    t = tensor[
            top_left[0]:bottom_right[0] + 1, 
            top_left[1]:bottom_right[1] + 1, 
            top_left[2]:bottom_right[2] + 1
            ]
    t = np.squeeze(t)
    # assert tf.rank(t) == 3

    final_shape = np.array([
                    config.IMAGE_DIMENSION_X * (1 + 2 * config.AUGMENTATION_SHIFT_MARGIN), 
                    config.IMAGE_DIMENSION_Y * (1 + 2 * config.AUGMENTATION_SHIFT_MARGIN), 
                    config.IMAGE_DIMENSION_Z * (1 + 2 * config.AUGMENTATION_SHIFT_MARGIN),
                    ], dtype=np.int32)
    # final_shape = tf.cast(final_shape, dtype = tf.int32)
    t = resize_3d.resize_3d_image(t, final_shape)
    return t


def cut_and_resize_including_pancreas(data: npt.NDArray[np.float32], mask: npt.NDArray[np.float32], top_left_percentage: float, bottom_right_percentage: float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    cutout and resize 3-d tensor with shape [w,h,d]
    cut overheads off from top_left to bottom_right + 1
    percentages are used to calculate from top_left and bottom_right

    for example, if top_left = [0,0,0] and bottom_right = [100,100,100]
    pancreas volume from [10,10,10] to [90,90,90]  
    if all percentages 0.1 (which is 10%) then
    the cut will be from [1,1,1] to [99,99,99]
    """

    start_prep = time.time()
    top_left_label_position = tf.reduce_min(tf.where(mask == 1), axis=0)
    bottom_right_label_position = tf.reduce_max(tf.where(mask == 1), axis=0)
    # random_offset_top_left = tf.random.uniform(shape = [3], minval = [0.0, 0.0, 0.0], maxval = tf.cast(top_left_label_position, dtype=tf.float32))
    top_left_offset = top_left_percentage * tf.cast(top_left_label_position, dtype=tf.float32)
    top_left_offset = tf.cast(top_left_offset, dtype = tf.int32)
    # random_offset_bottom_right = tf.random.uniform(shape = [3], minval = tf.cast(bottom_right_label_position, dtype=tf.float32), maxval = tf.cast(tf.shape(data), dtype=tf.float32))
    bottom_right_offset = tf.cast(tf.shape(data), dtype=tf.float32) - bottom_right_percentage * tf.cast(tf.shape(data) - tf.cast(bottom_right_label_position, dtype=tf.int32), dtype=tf.float32)
    bottom_right_offset = tf.cast(bottom_right_offset, dtype = tf.int32)
    finish_prep = time.time()

    if DEBUG_DATALOADER:
        print("\tpancreas shape:", (bottom_right_label_position - top_left_label_position).numpy())
        print("\ttop_left_label_position:", top_left_label_position.numpy(), "bottom_right_label_position:", bottom_right_label_position.numpy())
        print(f"\toriginal shape: {mask.shape}")
        print(f"\tpercentages: top_left: {top_left_percentage:.2f}, bottom_right: {bottom_right_percentage:.2f}")
        print("\toffset_top_left:", top_left_offset.numpy(), "bottom_right_offset:", bottom_right_offset.numpy())
        print("\tslice shape:", (bottom_right_offset - top_left_offset + 1).numpy())

    start_data = time.time()
    _data = __cutout_and_resize_tensor(data, top_left_offset, bottom_right_offset)
    finish_data = time.time()

    start_label = time.time()
    _label = __cutout_and_resize_tensor(mask, top_left_offset, bottom_right_offset)
    finish_label = time.time()

    if DEBUG_DATA_LOADING_PERFORMANCE:
        print(f"\tDATA_LOADING_PERFORMANCE: prep: {finish_prep - start_prep:.1f} data: {finish_data - start_data:.1f} label: {finish_label - start_label:.1f}")

    if DEBUG_DATALOADER:
        print("\t_data shape:", _data.shape, "_label shape:", _label.shape)

    return _data, _label

