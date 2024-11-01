import numpy as np
import nibabel as nib
from tools import resize_3d


def print_nifti_stat(nifti_img):
    data = nifti_img.get_fdata()
    print(nifti_img.affine)
    print("\taffine dtype:", nifti_img.affine.dtype)
    print("\tdata shape:", data.shape)
    print("\tdata dtype:", nifti_img.get_data_dtype())
    print("\tdata min/max:", np.min(data), np.max(data))


def main2():
    patient_id = "0017"
    # img = nib.load("c:\\docs\\src\\kt\\datasets\\ct-150\\labels\\label{}.nii".format(patient_id))
    img = nib.load("../ground_truth.nii")
    print("original:")
    print_nifti_stat(img)

    new_data = np.asarray(img.get_fdata(), dtype=np.uint8)
    resized_data = np.asarray(resize_3d.resize_3d_image(new_data, np.asarray(img.shape)).numpy(), dtype=np.uint8)

    resized_affine = img.affine
    resized_affine[2, 2] = -1
    img_to_save = nib.Nifti1Image(resized_data, resized_affine)
    nib.save(img_to_save, "test{}.nii".format(patient_id))


    img2 = nib.load("test{}.nii".format(patient_id))
    print("saved:")
    print_nifti_stat(img2)




if __name__ == "__main__":
    main2()

