import tensorflow as tf
import glob
import os
import pydicom
from tools import resize_3d
import numpy as np
import nibabel as nib
import tools.craft_network as craft_network
import config as config


class Predict:

    def __read_dcm_slices(self, dcm_folder):
        slices = [pydicom.dcmread(_) for _ in glob.glob(os.path.join(dcm_folder, "*.dcm"))]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        if len(slices) == 0:
            print("ERROR: no dcim files found")
        return slices

    def __get_pixel_data(self, dcm_slices):
        result = np.array([])
        if len(dcm_slices):
            result = np.stack([_.pixel_array for _ in dcm_slices], axis = -1)
        else:
            print("ERROR: dcim list is empty")

        return result

    def __preprocess_data(self, data):
        data = resize_3d.resize_3d_image(data, tf.constant(
            [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]))
        data = tf.cast(data, tf.float32)

        #
        # keep CT HU in range [pancreas HU]
        #
        data = tf.minimum(data, config.PANCREAS_MAX_HU)
        data = tf.maximum(data, config.PANCREAS_MIN_HU)

        #
        # scale CT to range [-1, 1]
        #
        data = (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data)) * 2 - 1

        data = data[tf.newaxis, ..., tf.newaxis]

        return data

    def __create_mask(self, data):
        mask = tf.argmax(data, axis = -1)
        mask = mask[..., tf.newaxis]
        return mask

    def __get_patient_position_from_first_frame(self, dcm_slices):
        min_number = dcm_slices[0][0x0020, 0x0013].value
        min_idx = 0
        for idx, _slice in enumerate(dcm_slices):
            if _slice[0x0020, 0x0013].value < min_number:
                min_number = _slice[0x0020, 0x0013].value
                min_idx = idx

        return dcm_slices[min_idx][0x0020, 0x0032].value

    def __get_affine_matrix(self, dcm_slices):
        result = []
        dcm_rows = dcm_slices[0][0x0028, 0x0010].value
        dcm_columns = dcm_slices[0][0x0028, 0x0011].value
        dcm_depth = len(dcm_slices)
        dcm_pixel_spacing = dcm_slices[0][0x0028, 0x0030].value
        dcm_instance_number_1 = dcm_slices[0][0x0020, 0x0013].value
        dcm_instance_number_2 = dcm_slices[1][0x0020, 0x0013].value
        dcm_patient_pos_1 = dcm_slices[0][0x0020, 0x0032].value[2]
        dcm_patient_pos_2 = dcm_slices[1][0x0020, 0x0032].value[2]
        dcm_slice_thickness = (dcm_patient_pos_2 - dcm_patient_pos_1) / (dcm_instance_number_2 - dcm_instance_number_1)
        dcm_patient_orientation = dcm_slices[0][0x0020, 0x0037].value
        dcm_patient_position = self.__get_patient_position_from_first_frame(dcm_slices)

        affine = np.zeros([4, 4])
        affine[0, 0] = dcm_patient_orientation[0] * dcm_pixel_spacing[0]
        affine[1, 0] = dcm_patient_orientation[1] * dcm_pixel_spacing[0]
        affine[2, 0] = dcm_patient_orientation[2] * dcm_pixel_spacing[0]

        affine[0, 1] = dcm_patient_orientation[3] * dcm_pixel_spacing[1]
        affine[1, 1] = dcm_patient_orientation[4] * dcm_pixel_spacing[1]
        affine[2, 1] = dcm_patient_orientation[5] * dcm_pixel_spacing[1]

        # --- inverse axes X and Y. This was found experimental way
        # --- could be wrong ... 
        affine[0, 0] = -affine[0, 0]
        affine[1, 0] = -affine[1, 0]
        affine[2, 0] = -affine[2, 0]

        affine[0, 1] = -affine[0, 1]
        affine[1, 1] = -affine[1, 1]
        affine[2, 1] = -affine[2, 1]

        affine[2, 2] = dcm_slice_thickness

        affine[0, 3] = dcm_patient_position[0]
        affine[1, 3] = dcm_patient_position[1]
        affine[2, 3] = dcm_patient_position[2]

        return affine

    def __resize_mask_to_dcm_shape(self, mask, dcm_slices):
        dcm_rows = dcm_slices[0][0x0028, 0x0010].value
        dcm_columns = dcm_slices[0][0x0028, 0x0011].value
        dcm_depth = len(dcm_slices)
        result = resize_3d.resize_3d_image(tf.squeeze(mask), tf.constant([dcm_rows, dcm_columns, dcm_depth]))
        result = result[tf.newaxis, ..., tf.newaxis]

        return result

    def __save_img_to_nifti(self, data, affine, result_file_name):
        img_to_save = nib.Nifti1Image(data, affine)
        nib.save(img_to_save, result_file_name)

    def __print_stat(self, data, title=""):
        print('-' * 100)
        if len(title):
            print(title)
        print("shape", data.shape)
        print("min/mean/max/sum {}/{:.2f}/{}/{}".format(tf.reduce_min(data),
                                                        tf.reduce_mean(tf.cast(data, dtype = tf.float32)),
                                                        tf.reduce_max(data), tf.reduce_sum(data)))

    def main(self, dcm_folder, result_file_name):
        dcm_slices = self.__read_dcm_slices(dcm_folder)
        raw_pixel_data = self.__get_pixel_data(dcm_slices)
        src_data = self.__preprocess_data(raw_pixel_data)

        model = craft_network.craft_network(config.MODEL_CHECKPOINT)
        # model = tf.keras.models.load_model("pancreas_segmentation_model.h5", compile=False)

        # model.summary()

        prediction = model(src_data)
        mask = self.__create_mask(prediction)
        mask = self.__resize_mask_to_dcm_shape(mask, dcm_slices)
        mask = tf.squeeze(mask)
        affine_matrix = self.__get_affine_matrix(dcm_slices)

        self.__save_img_to_nifti(np.asarray(mask.numpy(), dtype = np.uint8), affine_matrix, result_file_name)

        self.__print_stat(src_data, "src CT data")
        self.__print_stat(mask, "mask")
        print(affine_matrix)


if __name__ == "__main__":
    pred = Predict()
    pred.main("predict", "prediction.nii")
