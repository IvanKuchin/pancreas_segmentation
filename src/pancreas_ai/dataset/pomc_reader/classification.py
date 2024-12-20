import nibabel
import numpy as np
import numpy.typing as npt
import os
import tensorflow as tf

from . import interface
from . import dicom_nrrd

class Reader(interface.IReader):
    def __init__(self, config: dict):
        self.config = config

    def __read_ct_data(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        return dicom_nrrd.get_dicom_data(folder)


    def __read_nii(self, filename:str) -> npt.NDArray[np.int32]:
        nifti_data = nibabel.load(os.path.join(filename))
        return np.array(nifti_data.get_fdata(), dtype = int)


    def read_data(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        dicom, dicom_metadata = self.__read_ct_data(folder)
        if len(dicom.shape) == 0:
            return dicom, dicom_metadata
        
        self.mask = self.__read_nii(os.path.join(folder, self.config.CLASSIFICATION_SEGMENTATION_MASK_FILENAME))
        self.mask[self.mask == self.config.PANCREAS_ID_IN_MASK] = 1

        assert dicom.shape == self.mask.shape, "ERROR: DICOM and mask shapes are different"

        dicom[self.mask == 0] = 0

        return dicom, dicom_metadata
    
    def read_label(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        result = np.array([])
        fname = os.path.join(folder, self.config.CLASSIFICATION_LABEL_FILENAME)

        if not os.path.exists(fname):
            return  result
        
        with open(fname, "r") as f:
            result = np.array([int(f.readline())])

        return result, {}

    
    def check_before_processing(self, data, label, data_metadata, label_metadata):
        if data.shape[0] == 0:
            print("ERROR: data shape is incorrect (", data.shape, ")")
            return False
        
        if label.shape[0] == 0:
            print("ERROR: label shape is incorrect (", label.shape, ")")
            return False
        
        if np.mean(data) > 1000:
            print("ERROR: data mean(", np.mean(data), ") is too high. Probably probleam with reading DCIM data")
            return False

        return True

    def check_after_preprocessing(self, data: npt.NDArray[np.float32], label: npt.NDArray[np.float32]) -> bool:
        result = True

        # print("\tsanity check data: {}/{}/{}".format(np.min(data), np.mean(data), np.max(data)))
        # print("\tsanity check label: {}/{}/{}".format(np.min(label), np.mean(label), np.max(label)))

        if np.min(data) < self.config.MIN_DATA: # data scaled to range [-1, 1]
            result = False
            print("ERROR: (min(data) == {}) != -1".format(np.min(data)))
        if np.mean(data) == 0:
            result = False
            print("ERROR: (mean(data) == {}) == 0".format(np.mean(data)))
        if np.max(data) > self.config.MAX_DATA:
            result = False
            print("ERROR: (max(data) == {}) != 1".format(np.max(data)))

        if label[0] not in [0, 1]:
            result = False
            print("ERROR: label[0] must be 0 or 1, but it is", label[0])

        return result
    
    
    def rescale_if_needed(self, src_data: npt.NDArray[np.float32], label_data: npt.NDArray[np.float32], percentage: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        scaled_data, _ = dicom_nrrd.resacale_if_needed(src_data, self.mask, percentage, self.config)
        return scaled_data, label_data
    