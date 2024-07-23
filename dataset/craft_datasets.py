import glob
import time
import tensorflow as tf
import numpy as np
import borders
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d
import config as config


DEBUG_DATALOADER = False
DEBUG_DATA_LOADING_PERFORMANCE = False

def fname_to_patientid(fname_src:str):
    if DEBUG_DATALOADER:
        print("==========", fname_src)
    fname = fname_src.split(os.path.sep)[-1]
    patient_id = fname.split(sep = "_data")[0]
    return patient_id


def py_read_data_and_label(data_fname:str):
    if DEBUG_DATALOADER:
        print("data_fname:", data_fname)
    with np.load(data_fname) as content:
        data_label = content["arr_0"]
    data_array = data_label[0]
    label_array = data_label[1]
    return (tf.convert_to_tensor(data_array, dtype=tf.float32), tf.convert_to_tensor(label_array, dtype=tf.int32))


def read_data_and_label(patient_id:str, src_folder:str):
    """
    :type src_folder: basestring
    """
    if DEBUG_DATALOADER:
        print("src_folder:", src_folder, "patient_id:", patient_id)
    data_fname  = os.path.join(src_folder, patient_id)
    data_array, label_array = py_read_data_and_label(data_fname) #, Tout = (tf.float32, tf.int32))

    return data_array, label_array


class FileIterator:
    def __init__(self, folder):
        self.folder = folder
        self.file_list = glob.glob(os.path.join(folder, "*.npz"))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.file_list):
            data_file = self.file_list[self.index]
            self.index += 1
            return data_file
        else:
            raise StopIteration

class Array3d_read_and_resize:
    def __init__(self, folder):
        self.folder = folder

    def __call__(self):
        self.file_list = FileIterator(self.folder)
        for data_file in self.file_list:
            # print("file:", data_file)
            patient_id = fname_to_patientid(data_file)

            start_reading = time.time()
            data, label = read_data_and_label(patient_id, self.folder)
            finish_reading = time.time()

            start_resize = time.time()
            # data, label = borders.cut_and_resize_including_pancreas(data, label, np.random.rand(), np.random.rand())
            finish_resize = time.time()

            if DEBUG_DATA_LOADING_PERFORMANCE:
                print(f"\tDATA_LOADING_PERFORMANCE: reading time: {finish_reading - start_reading:.1f} resize time: {finish_resize - start_resize:.1f}")

            yield data, label


# def array3d_read_and_resize():
#     file_list = FileIterator(config.TFRECORD_FOLDER)
#     for data_file in file_list:
#         patient_id = fname_to_patientid(data_file)
#         data, label = read_data_and_label(patient_id, config.TFRECORD_FOLDER)

#         data, label = random_slice_including_pancreas(data, label)
#         yield data, label
    

def crop_to_shape(data, label):
    target_shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]
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


# add channel dimension axis=-1 
# (x, y, z) -> (x, y, z, channel)
def expand_dimension(data, label):
    data = data[..., tf.newaxis]
    label = label[..., tf.newaxis]
    return data, label


def craft_datasets(src_folder):
    result = None

    if os.path.isdir(src_folder):
        # d, l = array3d_read_and_resize()
                                    # .list_files(src_folder + "*_data.npy")\
                                    # .map(fname_to_patientid)\
                                    # .map(lambda patient_id: read_data_and_label(patient_id, src_folder))
                                    # .map(random_slice_including_pancreas)

        read_and_resize = Array3d_read_and_resize(src_folder)
        list_ds = tf.data.Dataset\
                                    .from_generator(read_and_resize, args=[], output_types=(tf.float16, tf.int16), output_shapes=([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z], [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]))\
                                    .map(random_flip)\
                                    .map(expand_dimension)\
                                    .batch(config.BATCH_SIZE)\
                                    # .prefetch(1)

        # total_number_of_entries = tf.data.experimental.cardinality(list_ds).numpy()
        # if total_number_of_entries == tf.data.experimental.UNKNOWN_CARDINALITY:
        #     total_number_of_entries = len(glob.glob(os.path.join(src_folder, "*_data.npy")))

        result = list_ds
    else:
        print("can't craft dataset, folder {} doesn't exists".format(src_folder))

    return result


class MeasureTime:
    def __init__(self, ds) -> None:
        self.iterator = iter(ds)

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        x = next(self.iterator)
        latency = time.time() - start
        return latency, x

def __run_through_data_wo_any_action(ds_train, ds_valid):
    for epoch in range(2):
        for i, (t, (data, label)) in enumerate(MeasureTime(ds_train)):
            print(f"train, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\t data shape: {data.shape}\tmean/std: {tf.reduce_mean(tf.cast( data, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast( data, dtype=tf.float32)).numpy():.2f}")
            print(f"train, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\tlabel shape: {label.shape}\tmean/std/sum: {tf.reduce_mean(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_sum(tf.cast(label, dtype=tf.float32)).numpy():.0f}")

        print("Valid ds:")
        for i, (t, (data, label)) in enumerate(MeasureTime(ds_valid)):
            print(f"valid, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\t data shape: {data.shape}\tmean/std: {tf.reduce_mean(tf.cast( data, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast( data, dtype=tf.float32)).numpy():.2f}")
            print(f"valid, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\tlabel shape: {label.shape}\tmean/std: {tf.reduce_mean(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast(label, dtype=tf.float32)).numpy():.2f}")



if __name__ == "__main__":
    train_ds = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"))
    valid_ds = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"))
    __run_through_data_wo_any_action(train_ds, valid_ds)
