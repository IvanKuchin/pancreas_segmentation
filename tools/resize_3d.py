import tensorflow as tf


# def resize_along_axis(tensor, new_size, axis_along):
#     # print("tensor shape: ", tensor.shape)
#     # print("axis:", axis_along)
#     unstacked = [_[tf.newaxis, ..., tf.newaxis] for _ in tf.unstack(tensor, axis = axis_along)]
#     unstacked = [tf.image.resize(_, new_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) for _ in unstacked]
#     unstacked = tf.squeeze(unstacked)
#     result = tf.stack(unstacked, axis = axis_along)
#     return result

def resize_along_axis(inp_tensor, new_size, axis_along):
    inp_tensor = tf.expand_dims(inp_tensor, axis = -1)
    restacked = tf.stack(tf.unstack(inp_tensor, axis = axis_along))
    resize = tf.image.resize(restacked, new_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resize = tf.cast(resize, dtype=inp_tensor.dtype)
    result = tf.stack(tf.unstack(resize, axis = 0), axis=axis_along)
    result = tf.squeeze(result, axis = -1)
    return result

def resize_3d_image(image, dimensions: tf.TensorSpec(shape=[3,], dtype=tf.int32)):
    assert dimensions.shape == (3,)
    zoomed_img = resize_along_axis(image, dimensions[:2], 2)
    zoomed_img = resize_along_axis(zoomed_img, dimensions[1:], 0)
    return zoomed_img
