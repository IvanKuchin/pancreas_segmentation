import unittest
import nibabel as nib
import glob
import os

nifti_folder = "c:\\docs\\src\\kt\\datasets\\ct-150\\labels\\"


class test_label_affine(unittest.TestCase):
    def get_affine_from_nifti_file(self, file_name):
        nifti = nib.load(file_name)
        affine = nifti.affine
        return affine

    def analyze(self, file_name, affine):
        if affine[2, 2] != -1:
            print("-----", file_name, "\n", affine)
        # sum = affine[1, 0] + affine[0, 1] + affine[2, 0] + affine[0, 2] + affine[2, 1] + affine[1, 2] + affine[3, 0] + affine[0, 3] + affine[3, 1] + affine[1, 3] + affine[3, 2] + affine[2, 3] + affine[3, 3]
        # if sum != 1:
        #     print("-----", file_name, "\n", affine)

    def analyze_translation(self, affine):
        result = True
        sum = affine[3, 0] + affine[0, 3] + affine[3, 1] + affine[1, 3] + affine[3, 2] + affine[2, 3] + affine[3, 3]
        if sum != 1:
            result = False

        return result

    def analyze_rotation(self, affine):
        result = True
        sum = affine[1, 0] + affine[0, 1] + affine[2, 0] + affine[0, 2] + affine[2, 1] + affine[1, 2]
        if sum != 0:
            result = False

        return result

    def test_is_translation_is_0(self):
        for file_name in glob.glob(os.path.join(nifti_folder, "label*.nii")):
            affine = self.get_affine_from_nifti_file(file_name)
            analysis_result = self.analyze_translation(affine)
            self.assertEqual(True, analysis_result, msg = "translation is non-zero in file {}".format(file_name))

    def test_is_rotation_is_0(self):
        for file_name in glob.glob(os.path.join(nifti_folder, "label*.nii")):
            affine = self.get_affine_from_nifti_file(file_name)
            analysis_result = self.analyze_rotation(affine)
            self.assertEqual(True, analysis_result, msg = "rotation is non-zero in file {}".format(file_name))


if __name__ == "__main__":
    unittest.main()
