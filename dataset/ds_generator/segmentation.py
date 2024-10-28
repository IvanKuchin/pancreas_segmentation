import tensorflow as tf

from ds_generator import interface


class Utils(interface.IUtils):
    def __init__(self, label_shape: list[int]):
        self.label_shape = label_shape
    
    def ds_label_shape(self) -> list[int]:
        return self.label_shape
    
    def expand_dimension(self, data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        data = data[..., tf.newaxis]
        label = label[..., tf.newaxis]
        return data, label
