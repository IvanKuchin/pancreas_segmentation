import tensorflow as tf
import numpy as np
import os

INPUT_DIMS = tf.constant([256, 256, 256])

def tfrecord_fname_to_patientid(fname_src):
    # print(fname_src)
    fname = tf.strings.split(fname_src, sep = os.path.sep)[-1]
    patient_id = tf.strings.split(fname, sep = ".")[0]
    return patient_id


def py_read_data_and_label(data_fname, data_label):
    data_array = np.load(data_fname.numpy())
    label_array = np.load(data_label.numpy())
    return (data_array.astype(np.float32), label_array.astype(np.int32))
    # return data_array.astype(np.float32)


def read_data_and_label(patient_id, src_folder):
    """

    :type src_folder: basestring
    """
    print("src_folder:", src_folder, "patient_id:", patient_id)
    data_fname = src_folder + patient_id + "_data.npy"
    label_fname = src_folder + patient_id + "_label.npy"
    data_array, label_array = tf.py_function(py_read_data_and_label, [data_fname, label_fname],
                                             Tout = (tf.float32, tf.int32))

    return data_array, label_array


def crop_to_shape(data, label):
    target_shape = INPUT_DIMS
    data_shape = tf.shape(data)
    random_range = data_shape - target_shape
    random_offset = tf.cast(
        tf.math.multiply(tf.random.uniform(shape = random_range.shape), tf.cast(random_range, dtype = tf.float32)),
        dtype = tf.int32)
    _data = data[random_offset[0]:random_offset[0] + target_shape[0],
            random_offset[1]:random_offset[1] + target_shape[1], random_offset[2]:random_offset[2] + target_shape[2],
            ...]
    _label = label[random_offset[0]:random_offset[0] + target_shape[0],
             random_offset[1]:random_offset[1] + target_shape[1], random_offset[2]:random_offset[2] + target_shape[2],
             ...]
    return _data, _label


def random_flip_along_axis(tensor1, tensor2, _axis):
    if tf.cast(tf.round(tf.random.uniform([1])), tf.bool):
        tensor1 = tf.reverse(tensor1, axis = [_axis])
        tensor2 = tf.reverse(tensor2, axis = [_axis])
    return tensor1, tensor2


def random_flip(data, label):
    data, label = random_flip_along_axis(data, label, 0)
    data, label = random_flip_along_axis(data, label, 1)
    data, label = random_flip_along_axis(data, label, 2)
    return data, label


def expand_dimension(data, label):
    data = data[..., tf.newaxis]
    label = label[..., tf.newaxis]

    data = tf.reshape(data, [256, 256, 256, 1])
    label = tf.reshape(label, [256, 256, 256, 1])

    return data, label


def craft_datasets(src_folder, ratio=0.2):
    list_ds = tf.data.Dataset.list_files(src_folder + "*.tfrecord").map(tfrecord_fname_to_patientid).map(
        lambda patient_id: read_data_and_label(patient_id, src_folder)).map(crop_to_shape).map(random_flip).map(expand_dimension).batch(2).prefetch(1)
    total_number_of_entries = tf.data.experimental.cardinality(list_ds).numpy()

    return list_ds.skip(total_number_of_entries * ratio), list_ds.take(total_number_of_entries * ratio)
