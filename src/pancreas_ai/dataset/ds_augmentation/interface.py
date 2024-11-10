import abc
import numpy as np
import numpy.typing as npt

class IAugment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_flip(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_rotate(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_resize(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError
