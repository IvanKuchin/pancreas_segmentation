import nibabel as nib
import glob
import os

nifti_folder = "c:\\docs\\src\\kt\\datasets\\ct-150\\labels\\"


def get_affine_from_nifti_file(file_name):
    nifti = nib.load(file_name)
    affine = nifti.affine
    return affine

def analyze(file_name, affine):
    if affine[2, 2] != -1:
        print("-----", file_name, "\n", affine)
    # sum = affine[1, 0] + affine[0, 1] + affine[2, 0] + affine[0, 2] + affine[2, 1] + affine[1, 2] + affine[3, 0] + affine[0, 3] + affine[3, 1] + affine[1, 3] + affine[3, 2] + affine[2, 3] + affine[3, 3]
    # if sum != 1:
    #     print("-----", file_name, "\n", affine)


def main():
    for file_name in glob.glob(os.path.join(nifti_folder, "label*.nii")):
        affine = get_affine_from_nifti_file(file_name)
        analyze(file_name, affine)


if __name__ == "__main__":
    main()


