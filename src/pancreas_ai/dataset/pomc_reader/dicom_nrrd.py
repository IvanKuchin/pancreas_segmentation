import glob
import pydicom
import nrrd
import os

import numpy as np
import numpy.typing as npt

import dataset.borders as borders

import src.pancreas_ai.config as config

import tensorflow as tf

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


def resacale_if_needed(src_data: npt.NDArray[np.float32], label_data: npt.NDArray[np.float32], percentage: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if config.IS_TILE == False:
        # scale data down to training size (augment border + resize)
        scaled_data, scaled_label = borders.cut_and_resize_including_pancreas(src_data, label_data, percentage/100, percentage/100)
    elif config.IS_TILE == True:
        scaled_data, scaled_label = tf.constant(src_data), tf.constant(label_data)
    else:
        print("ERROR: unknown IS_TILE value:", config.IS_TILE)
        
    return scaled_data, scaled_label