import numpy as np
import numpy.typing as npt
import abc

class IReader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read_data(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def read_label(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def check_before_processing(self, data: npt.NDArray[np.int32], label: npt.NDArray[np.int32], data_metadata: dict, label_metadata: dict) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def resacale_if_needed(self, src_data: npt.NDArray[np.float32], label_data: npt.NDArray[np.float32], percentage: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        raise NotImplementedError

    @abc.abstractmethod
    def check_after_preprocessing(self, data: npt.NDArray[np.float32], label: npt.NDArray[np.float32]) -> bool:
        raise NotImplementedError
    
