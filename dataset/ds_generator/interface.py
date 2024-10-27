import tensorflow as tf
import abc

class IUtils(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def ds_label_shape() -> list[int]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def expand_dimension(data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

