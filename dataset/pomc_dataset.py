import inspect
import time
import tensorflow as tf
import glob
import os
import sys
import re
import numpy as np
import pydicom
import nrrd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d
import tools.config as config

INPUT_DIMS = np.array([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z])
AUGMENT_SCALE_FACTOR = 0.1
AUGMENT_SCALED_DIMS = tf.cast(tf.constant(INPUT_DIMS, dtype = tf.float32) * (1 + AUGMENT_SCALE_FACTOR),
                              dtype = tf.int32).numpy()

PATIENTS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/test"
LABELS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/test"

DEBUG = True

class POMCDataset:
    def __init__(self, patients_src_folder, labels_src_folder, TFRECORD_FOLDER):
        self.patients_src_folder = patients_src_folder
        self.labels_src_folder = labels_src_folder
        self.TFRECORD_FOLDER = TFRECORD_FOLDER

    def get_patient_id_from_folder(self, folder):
        result = None
        m = re.search("(\\w+)$", folder)
        if m:
            result = m.group(1)
        return result

    @staticmethod
    def read_dicom_data_from_files(files):
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
            result = np.stack([_.pixel_array for _ in slices], axis = -1)
        else:
            print("ERROR: can't dcmread from files:", files)

        return result.astype(np.int32), metadata

    @staticmethod
    def read_nrrd_data_from_files(files):
        result = np.array([])
        metadata = {}

        data, header = nrrd.read(files[0])
        if data.shape[0]:
            result = data
            metadata["space origin"] = header["space origin"]
        else:
            print("ERROR: can't read nrrd data from files:", files)

        return result.astype(np.int32), metadata

    def get_dicom_data(self, folder):
        result = np.array([])
        file_list = glob.glob(os.path.join(folder, "*.dcm"))
        if len(file_list):
            result, metadata = self.read_dicom_data_from_files(file_list)
            if result.shape[0]:
                # --- nothing to do
                return result, metadata
            else:
                print("ERROR: reading DICOM data from files:", file_list)
        else:
            folders = glob.glob(os.path.join(folder, "*"))
            for f in folders:
                if os.path.isdir(f):
                    result = self.get_dicom_data(f)

        return result

    def get_nrrd_data(self, folder):
        result = np.array([])
        file_list = glob.glob(os.path.join(folder, "*.nrrd"))
        if len(file_list):
            result, metadata = self.read_nrrd_data_from_files(file_list)
            if result.shape[0]:
                # --- nothing to do
                return result, metadata
            else:
                print("ERROR: reading nrrd data from files:", file_list)
        else:
            folders = glob.glob(os.path.join(folder, "*"))
            for f in folders:
                if os.path.isdir(f):
                    result = self.get_nrrd_data(f)

        return result

    def _point_inside_box(self, min, max, point):
        return min[0]-1 <= point[0] <= max[0]+1 and min[1]-1 <= point[1] <= max[1]+1 and min[2]-1 <= point[2] <= max[2]+1

    def _points_close_to_each_other(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2)) < 1
    
    def _body_up_side_down(self, min):
        return min[2] > 0

    def consistency_check(self, data, label, data_metadata, label_metadata):
        if data.shape[0] == 0:
            print("ERROR: data shape is incorrect (", data.shape, ")")
            return False
        
        if label.shape[0] == 0:
            print("ERROR: label shape is incorrect (", label.shape, ")")
            return False
        
        if data.shape != label.shape:
            print("ERROR: data shape(", data.shape, ") is not equal to the label shape(", label.shape, ")")
            return False

        if not self._point_inside_box(data_metadata["min"], data_metadata["max"], label_metadata["space origin"]):
            print("ERROR: label space origin(", label_metadata["space origin"], ") is outside the data box(", data_metadata["min"], data_metadata["max"], ")")
            return False

        if not self._points_close_to_each_other(data_metadata["min"], label_metadata["space origin"]):
            print("ERROR: label space origin(", label_metadata["space origin"], ") is not close to the data first slice origin(", data_metadata["min"], ")")
            return False
        
        if self._body_up_side_down(data_metadata["min"]):
            print("ERROR: data is upside down")
            return False
        
        return True

    def print_statistic(self, tensor_old, tensor_new):
        print("\tshape:\t\t{} -> {}".format(tensor_old.shape, tensor_new.shape))
        print("\tmin/max:\t{}/{} -> {}/{}".format(np.min(tensor_old), np.max(tensor_old), np.min(tensor_new),
                                                np.max(tensor_new)))
        print("\tmean std:\t{:.3f} {:.3f} -> {:.3f} {:.3f}".format(np.mean(tensor_old), np.std(tensor_old), np.mean(tensor_new), np.std(tensor_new)))

    def preprocess_data(self, data, label):
        # zoom = AUGMENT_SCALED_DIMS / data.shape
        # data_zoomed = scipy.ndimage.interpolation.zoom(data, zoom, mode="nearest")
        # label_zoomed = scipy.ndimage.interpolation.zoom(label, zoom, mode="nearest")

        #
        # output minHU/maxHU of pancreas area
        #
        # gt_idx = label == 1
        # min_HU = np.min(data[gt_idx])
        # max_HU = np.max(data[gt_idx])
        # print("minTotal/minHU/maxHU/maxTotal: {}/{}/{}/{}".format(np.min(data), min_HU, max_HU, np.max(data)))


        #
        # Restrict CT voxel values to [pancreas HU], this will give wider range to pancreas,
        # compare to original data [pancreas HU]
        #
        data_idx1 = data <= config.PANCREAS_MIN_HU
        data_idx2 = data >= config.PANCREAS_MAX_HU

        data[data_idx1] = config.PANCREAS_MIN_HU
        data[data_idx2] = config.PANCREAS_MAX_HU

        #
        # Assign -1 to mask that is outside of pancreas HU
        #
        # label[data_idx1] = -1
        # label[data_idx2] = -1

        # data_zoomed = resize_3d.resize_3d_image(data, AUGMENT_SCALED_DIMS)
        # label_zoomed = resize_3d.resize_3d_image(label, AUGMENT_SCALED_DIMS)

        # self.print_statistic(label, label_zoomed)

        #
        # scale final data to [-1; 1] range, that should help with ReLU activation
        #
        data_processed = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1
        # if data_processed.shape != AUGMENT_SCALED_DIMS:
        #     print_error("wrong Z-axis dimensionality {} must be {}".format(data_processed.shape, AUGMENT_SCALED_DIMS))

        return data_processed, label

    def sanity_check_after_preprocessing(self, data, label):
        result = True

        # print("\tsanity check data: {}/{}/{}".format(np.min(data), np.mean(data), np.max(data)))
        # print("\tsanity check label: {}/{}/{}".format(np.min(label), np.mean(label), np.max(label)))

        if np.min(data) != -1: # data scaled to range [-1, 1]
            result = False
            print("ERROR: (min(data) == {}) != -1".format(np.min(data)))
        if np.mean(data) == 0:
            result = False
            print("ERROR: (mean(data) == {}) == 0".format(np.mean(data)))
        if np.max(data) != 1:
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
            if np.min(label) != 0:
                result = False
                print("ERROR: (min(label) == {}) != -1".format(np.min(label)))

        if np.mean(label) == 0:
            result = False
            print("ERROR: (mean(label) == {}) == 0".format(np.mean(label)))
        if np.max(label) != 1:
            result = False
            print("ERROR: (max(label) == {}) != 1".format(np.max(label)))

        return result

    def save_npy(self, patient_id, original_data, original_label, scaled_data, scaled_label):
        result = True
        scaled_data = np.cast[np.float32](scaled_data)
        scaled_label = np.cast[np.int8](scaled_label)
        np.save(self.TFRECORD_FOLDER + patient_id + "_data.npy", scaled_data)
        np.save(self.TFRECORD_FOLDER + patient_id + "_label.npy", scaled_label)

        return result

    def pickle_src_data(self):
        folder_list = glob.glob(os.path.join(self.patients_src_folder, "*"))

        for folder in folder_list:
            patient_id = self.get_patient_id_from_folder(folder)

            if len(patient_id) == 0:
                print("ERROR: identifying patient_id from folder:", folder)
                continue

            # --- read data about patient
            print("Read data about patient", patient_id)

            start_ts = time.time()
            src_data, src_metadata = self.get_dicom_data(folder)
            if len(src_data.shape) == 0:
                print("ERROR: can't read DICOM data from folder:", folder)
                continue
            print("\tRead DICOM data in {:.2f} sec".format(time.time() - start_ts))

            start_ts = time.time()
            label_data, label_metadata = self.get_nrrd_data(folder)
            if len(label_data.shape) == 0:
                print("ERROR: can't find nifti labels:", patient_id)
                continue
            print("\tRead NRRD labels in {:.2f} sec".format(time.time() - start_ts))

            if self.consistency_check(src_data, label_data, src_metadata, label_metadata) == False:
                print("ERROR: data & labels are not consistent patient_id:", patient_id)
                continue

            start_ts = time.time()
            scaled_src_data, scaled_label_data = self.preprocess_data(src_data, label_data)
            print("\tPreprocess data in {:.2f} sec".format(time.time() - start_ts))


            if DEBUG:
                print("\tData")
                self.print_statistic(src_data, scaled_src_data)
                print("\tLabel")
                self.print_statistic(label_data, scaled_label_data)


            if self.sanity_check_after_preprocessing(scaled_src_data, scaled_label_data) == False:
                print("ERROR: data or label failed sanity check")
                continue

            if self.save_npy(patient_id, src_data, label_data, scaled_src_data, scaled_label_data) == False:
                print("ERROR: can't save TFRecord patient id:", patient_id)


def main():
    pomc = POMCDataset(PATIENTS_SRC_FOLDER, LABELS_SRC_FOLDER, config.TFRECORD_FOLDER)
    pomc.pickle_src_data()
    return

if __name__ == "__main__":
    main()
