import tensorflow as tf


def resize_along_axis(tensor, new_size, axis_along):
    # print("tensor shape: ", tensor.shape)
    # print("axis:", axis_along)
    unstacked = [_[tf.newaxis, ..., tf.newaxis] for _ in tf.unstack(tensor, axis = axis_along)]
    unstacked = [tf.image.resize(_, new_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) for _ in unstacked]
    unstacked = tf.squeeze(unstacked)
    result = tf.stack(unstacked, axis = axis_along)
    return result


def resize_3d_image(image, dimensions: tf.TensorSpec(shape=[3,], dtype=tf.int32)):
    assert dimensions.shape == (3,)
    zoomed_img = resize_along_axis(image, dimensions[:2], 2)
    zoomed_img = resize_along_axis(zoomed_img, dimensions[1:], 0)
    return zoomed_img
