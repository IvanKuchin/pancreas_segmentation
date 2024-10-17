import tensorflow as tf
import dataset.borders as borders
import dataset.pomc_reader.interface as interface
import numpy.typing as npt
import numpy as np
import os
import glob
import pydicom
import nrrd
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d
import config as config

# Stored Values (SV) are the values stored in the image pixel data attribute.
# Representation value should be calculated as:
# Rescaled value = SV * Rescale Slope + Rescale Intercept
# https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281052
def __dcim_slice_stored_value_to_rescaled_value(slice):
    rescale_intercept = slice.RescaleIntercept if hasattr(slice, "RescaleIntercept") else 0
    rescale_slope = slice.RescaleSlope if hasattr(slice, "RescaleSlope") else 1
    return slice.pixel_array * rescale_slope + rescale_intercept


def __read_dicom_data_from_files(files):
    result = np.array([])
    metadata = {}

    slices = [pydicom.dcmread(file) for file in files]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    metadata["first slice origin"] = slices[0].ImagePositionPatient
    metadata["min"] = [
        min([_.ImagePositionPatient[0] for _ in slices]),
        min([_.ImagePositionPatient[1] for _ in slices]),
        min([_.ImagePositionPatient[2] for _ in slices]),
    ]
    metadata["max"] = [
        max([_.ImagePositionPatient[0] for _ in slices]),
        max([_.ImagePositionPatient[1] for _ in slices]),
        max([_.ImagePositionPatient[2] for _ in slices]),
    ]

    if len(slices):
        result = np.stack([__dcim_slice_stored_value_to_rescaled_value(_) for _ in slices], axis = -1)
        # result = np.stack([_.pixel_array for _ in slices], axis = -1)
    else:
        print("ERROR: can't dcmread from files:", files)

    return result.astype(np.int32), metadata


def __read_nrrd_data_from_files(files):
    result = np.array([])
    metadata = {}

    data, header = nrrd.read(files[0])
    if data.shape[0]:
        result = data
        metadata["space origin"] = header["space origin"]
    else:
        print("ERROR: can't read nrrd data from files:", files)

    return result.astype(np.int32), metadata


def get_dicom_data(folder: str) -> tuple[npt.NDArray[np.int32], dict]:
    result = np.array([])
    file_list = glob.glob(os.path.join(folder, "*.dcm"))
    if len(file_list):
        result, metadata = __read_dicom_data_from_files(file_list)
        if result.shape[0]:
            # --- normal return
            return result, metadata
        else:
            print("ERROR: reading DICOM data from files:", file_list)
    else:
        folders = glob.glob(os.path.join(folder, "*"))
        for f in folders:
            if os.path.isdir(f):
                result = get_dicom_data(f)

    # --- recursive return only
    return result


def get_nrrd_data(folder):
    result = np.array([])
    file_list = glob.glob(os.path.join(folder, "*.nrrd"))
    if len(file_list):
        result, metadata = __read_nrrd_data_from_files(file_list)
        if result.shape[0]:
            # --- normal return
            return result, metadata
        else:
            print("ERROR: reading nrrd data from files:", file_list)
    else:
        folders = glob.glob(os.path.join(folder, "*"))
        for f in folders:
            if os.path.isdir(f):
                result = get_nrrd_data(f)

    # --- recursive return only
    return result


class Reader(interface.IReader):
    def __init__(self):
        pass

    def read_data(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        return get_dicom_data(folder)
    
    def read_label(self, folder:str) -> tuple[npt.NDArray[np.int32], dict]:
        return get_nrrd_data(folder)
    
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

        # calculate min/max HU in pancreas area
        # gt_idx = label == 1
        # min_HU = np.min(data[gt_idx])
        # max_HU = np.max(data[gt_idx])
        # self.min_HU = min(self.min_HU, min_HU)
        # self.max_HU = max(self.max_HU, max_HU)

        # if min_HU < config.PANCREAS_MIN_HU or min_HU > config.PANCREAS_MAX_HU:
        #     print("ERROR: min HU(", min_HU, ") in pancreas area is out of range [", config.PANCREAS_MIN_HU, config.PANCREAS_MAX_HU, "]")
        #     return False
        
        # if max_HU < config.PANCREAS_MIN_HU or max_HU > config.PANCREAS_MAX_HU:
        #     print("ERROR: max HU(", max_HU, ") in pancreas area is out of range [", config.PANCREAS_MIN_HU, config.PANCREAS_MAX_HU, "]")
        #     return False
        
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
    
    def resacale_if_needed(self, src_data: npt.NDArray[np.float32], label_data: npt.NDArray[np.float32], percentage: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if config.IS_TILE == False:
            # scale data down to training size (augment border + resize)
            scaled_data, scaled_label = borders.cut_and_resize_including_pancreas(src_data, label_data, percentage/100, percentage/100)
        elif config.IS_TILE == True:
            scaled_data, scaled_label = tf.constant(src_data), tf.constant(label_data)
        else:
            print("ERROR: unknown IS_TILE value:", config.IS_TILE)
            
        return scaled_data, scaled_label