import tensorflow as tf
import numpy as np
import time
import os
import nibabel as nib

def print_model_weights(model):
    model.summary()
    for layer in model.layers:
        print(layer.name)
        print(layer.weights)


def main():
    model = tf.keras.models.load_model("pancreas_segmentation_model.h5")
    # print_model_weights(model)


def print_nifti_stat(nifti_img):
    data = nifti_img.get_fdata()
    print(nifti_img.affine)
    print("\taffine dtype:", nifti_img.affine.dtype)
    print("\tdata shape:", data.shape)
    print("\tdata dtype:", nifti_img.get_data_dtype())
    print("\tdata min/max:", np.min(data), np.max(data))


def main2():
    img = nib.load("c:\\docs\\src\\kt\\datasets\\ct-150\\labels\\label0001.nii")
    print("original:")
    print_nifti_stat(img)

    new_data = np.asarray(img.get_fdata(), dtype=np.uint8)
    img_to_save = nib.Nifti1Image(new_data, img.affine)
    nib.save(img_to_save, "test0001.nii")


    img2 = nib.load("c:\\Users\\ikuchin\\PycharmProjects\\ct_prediction\\test0001.nii")
    print("saved:")
    print_nifti_stat(img2)




if __name__ == "__main__":
    # main()
    main2()

