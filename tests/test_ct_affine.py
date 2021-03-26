import unittest
import nibabel as nib
import glob
import os
import pydicom
import numpy as np

patients_folder = "c:\\docs\\src\\kt\\datasets\\den\\data_by_series\\"


class test_label_affine(unittest.TestCase):
    def get_patient_orientations(self, file_name):
        _slice = pydicom.dcmread(file_name)
        return _slice[0x20, 0x37].value

    def get_patient_position(self, file_name):
        _slice = pydicom.dcmread(file_name)
        return _slice[0x20, 0x32].value


    def analyze(self, file_name, affine):
        if affine[2, 2] != -1:
            print("-----", file_name, "\n", affine)
        # sum = affine[1, 0] + affine[0, 1] + affine[2, 0] + affine[0, 2] + affine[2, 1] + affine[1, 2] + affine[3, 0] + affine[0, 3] + affine[3, 1] + affine[1, 3] + affine[3, 2] + affine[2, 3] + affine[3, 3]
        # if sum != 1:
        #     print("-----", file_name, "\n", affine)

    def analyze_translation(self, affine):
        """ analyze if PCS (patient coordinate system) located out of (0, 0, 0) """
        result = True
        _sum = affine[0] + affine[1] + affine[2]
        if _sum == 0:
            result = False

        return result

    def analyze_rotation(self, affine):
        """ analyze if patient oriented along two perpendicular axis """
        result = True
        _sum = np.sum(np.abs(np.asarray(affine)))
        if _sum != 2:
            result = False

        return result

    def test_is_translation_is_0(self):
        for patient_folder in glob.glob(os.path.join(patients_folder, "*")):
            for series_folder in glob.glob(os.path.join(patient_folder, "*")):
                for dcm_file in glob.glob(os.path.join(series_folder, "*.dcm")):
                    affine = self.get_patient_position(dcm_file)
                    analysis_result = self.analyze_translation(affine)
                    self.assertEqual(True, analysis_result, msg = "translation is non-zero in file {}".format(dcm_file))
                    break


    def test_is_rotation_is_0(self):
        for patient_folder in glob.glob(os.path.join(patients_folder, "*")):
            for series_folder in glob.glob(os.path.join(patient_folder, "*")):
                for dcm_file in glob.glob(os.path.join(series_folder, "*.dcm")):
                    affine = self.get_patient_orientations(dcm_file)
                    analysis_result = self.analyze_rotation(affine)
                    self.assertEqual(True, analysis_result, msg = "rotation is non-zero in file {}".format(dcm_file))
                    break


if __name__ == "__main__":
    unittest.main()
