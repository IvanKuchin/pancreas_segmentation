import tensorflow as tf
import dataset.borders as borders
import dataset.pomc_reader.interface as interface
import numpy.typing as npt
import numpy as np
import os
import inspect
import sys

from dataset.pomc_reader import dicom_nrrd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import src.pancreas_ai.config as config


class Reader(interface.IReader):
    def __init__(self):
        pass

    def read_data(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        return dicom_nrrd.get_dicom_data(folder)
    
    def read_label(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        return dicom_nrrd.get_nrrd_data(folder)
    
    def __point_inside_box(self, min, max, point):
        return min[0]-1 <= point[0] <= max[0]+1 and min[1]-1 <= point[1] <= max[1]+1 and min[2]-1 <= point[2] <= max[2]+1

    def __points_close_to_each_other(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2)) < 1

    def check_before_processing(self, data, label, data_metadata, label_metadata):
        if data.shape[0] == 0:
            print("ERROR: data shape is incorrect (", data.shape, ")")
            return False
        
        if label.shape[0] == 0:
            print("ERROR: label shape is incorrect (", label.shape, ")")
            return False
        
        if data.shape != label.shape:
            print("ERROR: data shape(", data.shape, ") is not equal to the label shape(", label.shape, ")")
            return False

        if not self.__point_inside_box(data_metadata["min"], data_metadata["max"], label_metadata["space origin"]):
            print("ERROR: label space origin(", label_metadata["space origin"], ") is outside the data box(", data_metadata["min"], data_metadata["max"], ")")
            return False

        if not self.__points_close_to_each_other(data_metadata["min"], label_metadata["space origin"]):
            print("ERROR: label space origin(", label_metadata["space origin"], ") is not close to the data first slice origin(", data_metadata["min"], ")")
            return False
        
        if np.mean(data) > 1000:
            print("ERROR: data mean(", np.mean(data), ") is too high. Probably probleam with reading DCIM data")
            return False

        return True

    def check_after_preprocessing(self, data: npt.NDArray[np.float32], label: npt.NDArray[np.float32]) -> bool:
        result = True

        # print("\tsanity check data: {}/{}/{}".format(np.min(data), np.mean(data), np.max(data)))
        # print("\tsanity check label: {}/{}/{}".format(np.min(label), np.mean(label), np.max(label)))

        if np.min(data) != config.MIN_DATA: # data scaled to range [-1, 1]
            result = False
            print("ERROR: (min(data) == {}) != -1".format(np.min(data)))
        if np.mean(data) == 0:
            result = False
            print("ERROR: (mean(data) == {}) == 0".format(np.mean(data)))
        if np.max(data) != config.MAX_DATA:
            result = False
            print("ERROR: (max(data) == {}) != 1".format(np.max(data)))

        # labels must be
        # -1, if HU in [pancreas HU-s]
        #  0 - background
        #  1 - pancreas
        if config.PANCREAS_MIN_HU > 2000:
            if np.min(label) != -1:
                result = False
                print("ERROR: (min(label) == {}) != -1".format(np.min(label)))
        else:
            if np.min(label) != config.MIN_LABEL:
                result = False
                print("ERROR: (min(label) == {}) != -1".format(np.min(label)))

        if np.mean(label) == 0:
            result = False
            print("ERROR: (mean(label) == {}) == 0".format(np.mean(label)))
        if np.max(label) != config.MAX_LABEL:
            result = False
            print("ERROR: (max(label) == {}) != 1".format(np.max(label)))

        return result
    
    def rescale_if_needed(self, src_data: npt.NDArray[np.float32], label_data: npt.NDArray[np.float32], percentage: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return dicom_nrrd.resacale_if_needed(src_data, label_data, percentage)
    