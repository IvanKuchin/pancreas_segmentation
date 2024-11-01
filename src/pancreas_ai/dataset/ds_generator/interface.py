import tensorflow as tf
import abc

class IUtils(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def ds_label_shape(self) -> list[int]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def expand_dimension(self, data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        # add channel dimension axis=-1 (if needed)
        # (x, y, z) -> (x, y, z, channel)
        """
        raise NotImplementedError

