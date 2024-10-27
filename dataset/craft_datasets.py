import glob
import time
import tensorflow as tf
import numpy as np
import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from ds_generator.factory import ds_generator_factory
from dataset.ds_augmentation.factory import augment_factory
import config as config


DEBUG_DATALOADER = False
DEBUG_DATA_LOADING_PERFORMANCE = False

def fname_from_full_path(fname_src:str):
    if DEBUG_DATALOADER:
        print("fname_to_patientid: ", fname_src)
    fname = fname_src.split(os.path.sep)[-1]
    return fname


def read_data_and_label(patient_id:str, src_folder:str) -> tuple[np.ndarray, np.ndarray]:
    """
    read data and label from src_folder/patient_id.npz
    """
    if DEBUG_DATALOADER:
        print("read_data_and_label: src_folder:", src_folder, "patient_id:", patient_id)

    with open(os.path.join(src_folder, patient_id), "rb") as f:
        data_array = np.load(f, allow_pickle=True)["a"]
        label_array = np.load(f, allow_pickle=True)["b"]


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
        self.augment = augment_factory(config.TASK_TYPE)

    def __call__(self):
        self.file_list = FileIterator(self.folder)
        for data_file in self.file_list:
            if DEBUG_DATALOADER:
                print("__call__: file:", data_file)

            if data_file.find("_data") == -1:
                continue

            patient_id = fname_from_full_path(data_file)

            start_reading = time.time()
            data, label = read_data_and_label(patient_id, self.folder)
            finish_reading = time.time()

            data, label = self.augment.random_crop(data, label, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z)

            start_resize = time.time()
            # data, label = borders.cut_and_resize_including_pancreas(data, label, np.random.rand(), np.random.rand())
            finish_resize = time.time()

            if DEBUG_DATA_LOADING_PERFORMANCE:
                print(f"\tDATA_LOADING_PERFORMANCE: reading time: {finish_reading - start_reading:.1f} resize time: {finish_resize - start_resize:.1f}")

            yield tf.convert_to_tensor(data, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int8)


# def array3d_read_and_resize():
#     file_list = FileIterator(config.TFRECORD_FOLDER)
#     for data_file in file_list:
#         patient_id = fname_to_patientid(data_file)
#         data, label = read_data_and_label(patient_id, config.TFRECORD_FOLDER)

#         data, label = random_slice_including_pancreas(data, label)
#         yield data, label
    

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
def expand_dimension(data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    if config.TASK_TYPE == "segmentation":
        data = data[..., tf.newaxis]
        label = label[..., tf.newaxis]
    elif config.TASK_TYPE == "classification":
        data = data[..., tf.newaxis]
    else:
        raise ValueError("Unknown TASK_TYPE")
    return data, label


def ds_label_shape() -> list[int]:
    shape = []
    if config.TASK_TYPE == "segmentation":
        shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]
    elif config.TASK_TYPE == "classification":
        shape = [config.LABEL_CLASSIFICATION_DIMENSION]
    else:
        raise ValueError("Unknown TASK_TYPE")

    return shape

def craft_datasets(src_folder):
    result = None

    if os.path.isdir(src_folder):
        # d, l = array3d_read_and_resize()
                                    # .list_files(src_folder + "*_data.npy")\
                                    # .map(fname_to_patientid)\
                                    # .map(lambda patient_id: read_data_and_label(patient_id, src_folder))
                                    # .map(random_slice_including_pancreas)

        utils = ds_generator_factory(config)

        read_and_resize = Array3d_read_and_resize(src_folder)
        list_ds = tf.data.Dataset\
                                    .from_generator(
                                        read_and_resize, 
                                        args=[], 
                                        output_signature=(
                                            tf.TensorSpec(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z], dtype = tf.float32),
                                            tf.TensorSpec(shape = ds_label_shape(), dtype = tf.int32)
                                        ),
                                        # output_types=(tf.float16, tf.int16), 
                                        # output_shapes=([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z], [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z])
                                    )\
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

def __pancreas_histogram(data, label):
    print(f"\tdata  histogram1: {np.histogram(data, bins = 10)}")
    print(f"\tlabel histogram2: {np.histogram(label, bins = 10)}")


def __run_through_data_wo_any_action(ds_train, ds_valid):
    for epoch in range(2):
        for i, (t, (data, label)) in enumerate(MeasureTime(ds_train)):
            print(f"train, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\t data shape: {data.shape}\tmean/std: {tf.reduce_mean(tf.cast( data, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast( data, dtype=tf.float32)).numpy():.2f}")
            print(f"train, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\tlabel shape: {label.shape}\tmean/std/sum: {tf.reduce_mean(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_sum(tf.cast(label, dtype=tf.float32)).numpy():.0f}")
            __pancreas_histogram(data, label)

        print("Valid ds:")
        for i, (t, (data, label)) in enumerate(MeasureTime(ds_valid)):
            print(f"valid, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\t data shape: {data.shape}\tmean/std: {tf.reduce_mean(tf.cast( data, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast( data, dtype=tf.float32)).numpy():.2f}")
            print(f"valid, epoch/batch {epoch}/{i:02d},\tlatency {t:.1f}\tlabel shape: {label.shape}\tmean/std: {tf.reduce_mean(tf.cast(label, dtype=tf.float32)).numpy():.2f}/{tf.math.reduce_std(tf.cast(label, dtype=tf.float32)).numpy():.2f}")



if __name__ == "__main__":
    # read_and_resize = Array3d_read_and_resize(os.path.join(config.TFRECORD_FOLDER, "train"))
    # item1 = next(read_and_resize())
    # print("item1:", item1[0].shape, item1[1].shape)

    train_ds = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"))
    valid_ds = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"))
    __run_through_data_wo_any_action(train_ds, valid_ds)
