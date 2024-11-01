import tensorflow as tf
import glob
import os
import pydicom
from dataset.thickness.factory import ThicknessFactory
from tools import resize_3d
import numpy as np
import nibabel as nib
import tools.craft_network as craft_network
import src.pancreas_ai.config as config
from tools.predict.factory import PredictFactory
from tools.predict.predict_no_tile import PredictNoTile
from tools.predict.predict_tile import PredictTile


class Predict:

    def __read_dcm_slices(self, dcm_folder):
        slices = [pydicom.dcmread(_) for _ in glob.glob(os.path.join(dcm_folder, "*.dcm"))]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        if len(slices) == 0:
            print("ERROR: no dcim files found")
        return slices

    # Stored Values (SV) are the values stored in the image pixel data attribute.
    # Representation value should be calculated as:
    # Rescaled value = SV * Rescale Slope + Rescale Intercept
    # https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281052
    def __dcim_slice_stored_value_to_rescaled_value(self, slice):
        rescale_intercept = slice.RescaleIntercept if hasattr(slice, "RescaleIntercept") else 0
        rescale_slope = slice.RescaleSlope if hasattr(slice, "RescaleSlope") else 1
        return slice.pixel_array * rescale_slope + rescale_intercept

    def __get_pixel_data(self, dcm_slices):
        result = np.array([])
        if len(dcm_slices):
            result = np.stack([self.__dcim_slice_stored_value_to_rescaled_value(_) for _ in dcm_slices], axis = -1)
        else:
            print("ERROR: dcim list is empty")

        return result

    def __preprocess_data(self, data):
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

    # def __create_segmentation(self, data):
    #     mask = tf.argmax(data, axis = -1)
    #     mask = mask[..., tf.newaxis]
    #     return mask

    def __get_patient_position_from_first_frame(self, dcm_slices):
        min_number = dcm_slices[0][0x0020, 0x0013].value
        min_idx = 0
        for idx, _slice in enumerate(dcm_slices):
            if _slice[0x0020, 0x0013].value < min_number:
                min_number = _slice[0x0020, 0x0013].value
                min_idx = idx

        return dcm_slices[min_idx][0x0020, 0x0032].value

    def __get_metadata(self, dcm_slices):
        result = []
        dcm_rows = dcm_slices[0][0x0028, 0x0010].value
        dcm_columns = dcm_slices[0][0x0028, 0x0011].value
        dcm_depth = len(dcm_slices)
        dcm_pixel_spacing = dcm_slices[0][0x0028, 0x0030].value

        thickness_func = ThicknessFactory(config.THICKNESS)
        dcm_slice_thickness_0 = thickness_func(dcm_slices, 0)
        dcm_slice_thickness_1 = thickness_func(dcm_slices, 1)
        dcm_slice_thickness_2 = thickness_func(dcm_slices, 2)

        dcm_patient_orientation = dcm_slices[0][0x0020, 0x0037].value
        dcm_patient_position = self.__get_patient_position_from_first_frame(dcm_slices)

        affine = np.zeros([4, 4])
        affine[0, 0] = dcm_patient_orientation[0] * dcm_pixel_spacing[0]
        affine[1, 0] = dcm_patient_orientation[1] * dcm_pixel_spacing[0]
        affine[2, 0] = dcm_patient_orientation[2] * dcm_pixel_spacing[0]

        affine[0, 1] = dcm_patient_orientation[3] * dcm_pixel_spacing[1]
        affine[1, 1] = dcm_patient_orientation[4] * dcm_pixel_spacing[1]
        affine[2, 1] = dcm_patient_orientation[5] * dcm_pixel_spacing[1]

        affine[0, 3] = dcm_patient_position[0]
        affine[1, 3] = dcm_patient_position[1]
        affine[2, 3] = dcm_patient_position[2]

        affine[2, 0] = dcm_slice_thickness_0
        affine[2, 1] = dcm_slice_thickness_1
        affine[2, 2] = dcm_slice_thickness_2

        # --- inverse axes X and Y. This was found experimental way
        # --- could be wrong ... 
        affine[0, 0] = -affine[0, 0]
        affine[1, 0] = -affine[1, 0]
        affine[2, 0] = -affine[2, 0]

        affine[0, 1] = -affine[0, 1]
        affine[1, 1] = -affine[1, 1]
        affine[2, 1] = -affine[2, 1]

        affine[0, 2] = -affine[0, 2]
        affine[1, 2] = -affine[1, 2]
        affine[2, 2] = -affine[2, 2]

        affine[0, 3] = -affine[0, 3]
        affine[1, 3] = -affine[1, 3]

        return {"affine": affine, "spacing": dcm_pixel_spacing, "dim": [dcm_rows, dcm_columns, dcm_depth]}

    # def __scale_up(self, mask, dcm_slices):
    #     dcm_rows = dcm_slices[0][0x0028, 0x0010].value
    #     dcm_columns = dcm_slices[0][0x0028, 0x0011].value
    #     dcm_depth = len(dcm_slices)
    #     result = resize_3d.resize_3d_image(tf.squeeze(mask), tf.constant([dcm_rows, dcm_columns, dcm_depth]))

    #     return result

    # def __scale_down(self, data):
    #     if config.IS_TILE == True:
    #         data = tf.cast(data, tf.float32)
    #     elif config.IS_TILE == False:
    #         data = resize_3d.resize_3d_image(data, tf.constant(
    #             [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]))
    #         data = tf.cast(data, tf.float32)
    #     else:
    #         raise ValueError("Unknown IS_TILE value: " + config.IS_TILE)
    #     return data

    def __save_img_to_nifti(self, data, meta, result_file_name):
        # TODO: add meta information
        # affine = meta['affine'][0].cpu().numpy()
        # pixdim = meta['pixdim'][0].cpu().numpy()
        # dim    = meta['dim'][0].cpu().numpy()

        # img = nib.Nifti1Image(input_nii_array, affine=affine)
        # img.header['dim'] = dim
        # img.header['pixdim'] = pixdim

        img = nib.Nifti1Image(data, meta["affine"])
        # img.header['dim'] = meta["dim"]
        # img.header['pixdim'] = meta["spacing"]

        nib.save(img, result_file_name)

    def __print_stat(self, data, title=""):
        if len(title):
            print('-' * 25, title, '-' * 25)
        print("shape", data.shape)
        print("min/mean/max/sum {}/{:.5f}/{}/{}".format(tf.reduce_min(data),
                                                        tf.reduce_mean(tf.cast(data, dtype = tf.float32)),
                                                        tf.reduce_max(data), tf.reduce_sum(data)))


    def main(self, dcm_folder, result_file_name):
        model = craft_network.craft_network(config.MODEL_CHECKPOINT)
        # model.summary()

        predict_class = PredictFactory()("tile" if config.IS_TILE else "no_tile")
        predict_obj = predict_class(model)

        dcm_slices = self.__read_dcm_slices(dcm_folder)
        raw_pixel_data = self.__get_pixel_data(dcm_slices)
        scaled_data = predict_obj.scale_down(raw_pixel_data)
        # scaled_data = self.__scale_down(raw_pixel_data)
        src_data = self.__preprocess_data(scaled_data)

        mask = predict_obj.predict(src_data)

        # mask = self.__create_segmentation(mask)
        # mask = tf.squeeze(mask)
        mask = predict_obj.scale_up(mask)
        # mask = self.__scale_up(mask, dcm_slices)
        metadata = self.__get_metadata(dcm_slices)

        self.__save_img_to_nifti(np.asarray(mask.numpy(), dtype = np.uint8), metadata, result_file_name)

        self.__print_stat(src_data, "src CT data")
        self.__print_stat(mask, "mask")
        print(metadata)


if __name__ == "__main__":
    pred = Predict()
    pred.main("predict", "prediction.nii")
