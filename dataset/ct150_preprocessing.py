import tensorflow as tf
import numpy as np
import pydicom
import glob
import os
import re
import nibabel
from tools import resize_3d
import tools.config as config

PATIENTS_SRC_FOLDER = "/docs/src/kt/datasets/ct-150/data/"
LABELS_SRC_FOLDER = "/docs/src/kt/datasets/ct-150/labels/"
INPUT_DIMS = np.array([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z])
AUGMENT_SCALE_FACTOR = 0.1
AUGMENT_SCALED_DIMS = tf.cast(tf.constant(INPUT_DIMS, dtype = tf.float32) * (1 + AUGMENT_SCALE_FACTOR),
                              dtype = tf.int32).numpy()


def print_error(text_to_print):
    print("ERROR:", text_to_print)


def print_debug(text_to_print):
    print(text_to_print)


class CT150:
    def __init__(self, patients_src_folder, labels_src_folder, TFRECORD_FOLDER):
        self.patients_src_folder = patients_src_folder
        self.labels_src_folder = labels_src_folder
        self.TFRECORD_FOLDER = TFRECORD_FOLDER

    def GetPatientIDFromFolder(self, folder):
        result = None
        m = re.search("PANCREAS_(\\d+)", folder)
        if m:
            result = m.group(1)
        return result

    @staticmethod
    def read_dicom_data_from_files(files):
        result = np.array([])

        slices = [pydicom.dcmread(file) for file in files]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        if len(slices):
            result = np.stack([_.pixel_array for _ in slices], axis = -1)
        else:
            print("ERROR: can't dcmread from files:", files)

        return result

    def GetDICOMData(self, folder):
        result = np.array([])
        file_list = glob.glob(os.path.join(folder, "*.dcm"))
        if len(file_list):
            result = self.read_dicom_data_from_files(file_list)
            if result.shape[0]:
                # --- nothing to do
                a = 1
            else:
                print("ERROR: reading DICOM data from files:", file_list)
        else:
            folders = glob.glob(os.path.join(folder, "*"))
            if len(folders):
                result = self.GetDICOMData(folders[0])

        return result

    def GetNiftiData(self, patient_id):
        nifti_data = nibabel.load(os.path.join(self.labels_src_folder, "label" + patient_id + ".nii"))
        return np.array(nifti_data.get_fdata(), dtype = int)

    def ConsistencyCheck(self, data, label):
        result = False

        if data.shape[0]:
            if label.shape[0]:
                if data.shape == label.shape:
                    result = True
                else:
                    print("ERROR: data shape is not equal to label shape")
            else:
                print("ERROR: label shape is incorrect (", label.shape, ")")
        else:
            print("ERROR: data shape is incorrect (", data.shape, ")")

        return result

    def save_tfrecords(self, patient_id, original_data, original_label, scaled_data, scaled_label):
        result = True

        # print("\tshape:", original_data.shape, "->", scaled_data.shape)

        if not os.path.isdir(self.TFRECORD_FOLDER):
            os.mkdir(self.TFRECORD_FOLDER)

        with tf.io.TFRecordWriter(os.path.join(self.TFRECORD_FOLDER, patient_id + ".tfrecord")) as f:
            feature = {
                "original_shape": tf.train.Feature(int64_list = tf.train.Int64List(value = original_data.shape)),
                "scaled_shape": tf.train.Feature(int64_list = tf.train.Int64List(value = scaled_data.shape)),
                # "scaled_data": tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(scaled_data, [-1]))),
                # "scaled_label": tf.train.Feature(int64_list = tf.train.Int64List(value = tf.reshape(tf.constant(scaled_label), [-1])))
            }

            example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
            f.write(example_proto.SerializeToString())
            # f.write(tf.io.serialize_tensor(scaled_data))

        np.save(config.TFRECORD_FOLDER + patient_id + "_data.npy", scaled_data)
        np.save(config.TFRECORD_FOLDER + patient_id + "_label.npy", scaled_label)

        return result

    # def resize_along_axis(self, tensor, new_size, axis_along):
    #     # print("tensor shape: ", tensor.shape)
    #     unstacked = [_[tf.newaxis, ..., tf.newaxis] for _ in tf.unstack(tensor, axis = axis_along)]
    #     unstacked = [tf.image.resize(_, new_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) for _ in unstacked]
    #     result = tf.squeeze(tf.stack(unstacked, axis = axis_along))
    #     return result
    #
    # def resize_3d_image(self, image, dimensions):
    #     assert dimensions.shape == (3,)
    #     zoomed_img = self.resize_along_axis(image, dimensions[:2], 2)
    #     zoomed_img = self.resize_along_axis(zoomed_img, dimensions[1:], 0)
    #     return zoomed_img
    #
    #
    def print_statistic(self, tensor_old, tensor_new):
        print("\tshape) {}".format(tensor_new.shape))
        print("\tmin/max {}/{} -> {}/{}".format(np.min(tensor_old), np.max(tensor_old), np.min(tensor_new),
                                                np.max(tensor_new)))
        print("\t\thistogram origin: {}".format(np.histogram(tensor_old, bins = 10)))
        print("\t\thistogram zoomed: {}".format(np.histogram(tensor_new, bins = 10)))

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
        label[data_idx1] = -1
        label[data_idx2] = -1

        data_zoomed = resize_3d.resize_3d_image(data, AUGMENT_SCALED_DIMS)
        label_zoomed = resize_3d.resize_3d_image(label, AUGMENT_SCALED_DIMS)

        # self.print_statistic(label, label_zoomed)

        #
        # scale final data to [-1; 1] range, that should help with ReLU activation
        #
        data_processed = (data_zoomed - np.min(data_zoomed)) / (np.max(data_zoomed) - np.min(data_zoomed)) * 2 - 1
        if data_processed.shape != AUGMENT_SCALED_DIMS:
            print_error("wrong Z-axis dimensionality {} must be {}".format(data_processed.shape, AUGMENT_SCALED_DIMS))

        return data_processed, label_zoomed

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
        # -1, if HU <-100 or >200
        #  0 - background
        #  1 - pancreas
        if np.min(label) != -1:
            result = False
            print("ERROR: (min(label) == {}) != -1".format(np.min(label)))
        if np.mean(label) == 0:
            result = False
            print("ERROR: (mean(label) == {}) == 0".format(np.mean(label)))
        if np.max(label) != 1:
            result = False
            print("ERROR: (max(label) == {}) != 1".format(np.max(label)))

        return result

    def read_src_data_and_labels_save_as_tfrecords(self):
        folder_list = glob.glob(os.path.join(self.patients_src_folder, "*"))

        for folder in folder_list:
            patient_id = self.GetPatientIDFromFolder(folder)
            if patient_id:
                # --- read data about patient
                print("Read data about patient", patient_id)

                src_data = self.GetDICOMData(folder)
                if src_data.shape[0]:
                    # print("\tread from dicom files:", src_data.shape)
                    pass
                else:
                    print("ERROR: can't read DICOM data from folder:", folder)

                label_data = self.GetNiftiData(patient_id)
                if label_data.shape[0]:
                    # print("label mean:", label_data.get_fdata().mean())
                    pass
                else:
                    print("ERROR: can't find nifti labels:", patient_id)

                if self.ConsistencyCheck(src_data, label_data):
                    # print("\tdata & labels are consistent")

                    scaled_src_data, scaled_label_data = self.preprocess_data(src_data, label_data)

                    if self.sanity_check_after_preprocessing(scaled_src_data, scaled_label_data):
                        # pass
                        if self.save_tfrecords(patient_id, src_data, label_data, scaled_src_data, scaled_label_data):
                            pass
                        else:
                            print("ERROR: can't save TFRecord patient id:", patient_id)
                    else:
                        print("ERROR: data or label failed sanity check")
                else:
                    print("ERROR: data & labels are not consistent")
            else:
                print("ERROR: identifying patient_id from folder:", folder)


def main():
    ct150 = CT150(PATIENTS_SRC_FOLDER, LABELS_SRC_FOLDER, config.TFRECORD_FOLDER)
    ct150.read_src_data_and_labels_save_as_tfrecords()


def main2():
    arr = np.random.randint(10, size = (500, 50, 500))
    print(arr.shape)
    np.save(config.TFRECORD_FOLDER + "/1.npy", arr)


def main3():
    arr2 = np.load(config.TFRECORD_FOLDER + "/0001_data.npy")
    print(arr2.shape)


def main4():
    r1 = np.arange(-1024, 2048)
    norm1 = (r1 - np.min(r1)) / (np.max(r1) - np.min(r1))
    print("{}/{}".format(np.min(norm1), np.max(norm1)))


if __name__ == "__main__":
    main()
    # main2()
    # main3()
    # main4()
