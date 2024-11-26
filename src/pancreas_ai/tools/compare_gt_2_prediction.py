import numpy as np
import nibabel as nib
import tensorflow as tf


def print_nifti_stat(nifti_img):
    data = nifti_img.get_fdata()
    print(nifti_img.affine)
    print("\taffine dtype:", nifti_img.affine.dtype)
    print("\tdata shape:", data.shape)
    print("\tdata dtype:", nifti_img.get_data_dtype())
    print("\tdata min/max:", np.min(data), np.max(data))
    print("\tdata sum/total:", np.sum(data), data.reshape([-1]).shape[0])

def get_metrics(gt, pred):
    gt = tf.constant(gt)
    pred = tf.constant(pred)

    m1 = tf.keras.metrics.Precision()
    m1.reset_state()
    m1.update_state(gt, pred)
    precision = m1.result()

    m1 = tf.keras.metrics.Recall()
    m1.reset_state()
    m1.update_state(gt, pred)
    recall = m1.result()

    f1 = 2*precision*recall/(precision + recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("f1:", f1)

    return precision, recall, f1

def compare(gt_nifti_file, prediction_nifti_file):
    gt = nib.load(gt_nifti_file)
    gt_data = np.asarray(gt.get_fdata(), dtype=np.uint8)
    print("ground truth:")
    print_nifti_stat(gt)


    pred = nib.load(prediction_nifti_file)
    pred_data = np.asarray(pred.get_fdata(), dtype=np.uint8)
    print("prediction:")
    print_nifti_stat(pred)

    return get_metrics(gt_data, pred_data)


if __name__ == "__main__":
    compare("../ground_truth.nii", "../prediction.nii")

