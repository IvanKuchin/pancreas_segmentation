import tensorflow as tf

from ds_generator import interface


class Utils(interface.IUtils):
    def __init__(self):
        pass
    
    def ds_label_shape() -> list[int]:
        return [256, 256, 1]
    
    def expand_dimension(data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        data = data[..., tf.newaxis]
        label = label[..., tf.newaxis]
        return data, label
