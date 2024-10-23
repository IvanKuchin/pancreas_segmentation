import numpy as np
import numpy.typing as npt
import abc

class IAugment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random_crop(self, data:npt.NDArray[np.float32], mask:npt.NDArray[np.float32], inside_width: int, inside_height: int, inside_depth: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        raise NotImplementedError
    
