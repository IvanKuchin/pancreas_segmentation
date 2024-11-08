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

from .ds_generator.factory import ds_generator_factory
from .ds_augmentation.factory import augment_factory
from pancreas_ai.tools import resize_3d
import config


DEBUG_DATALOADER = False
DEBUG_DATA_LOADING_PERFORMANCE = True

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

            start_flip = time.time()
            data, label = self.augment.random_crop(data, label, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z)
            finish_flip = time.time()

            start_resize = time.time()
            # data = resize_3d.resize_3d_image(data, np.array([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]))
            finish_resize = time.time()

            start_rotate = time.time()
            # data, label = self.augment.rotate(data, label, np.array([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z]))
            finish_rotate = time.time()

            start_flip = time.time()
            data, label = self.augment.random_flip(data, label)
            finish_flip = time.time()

            if DEBUG_DATA_LOADING_PERFORMANCE:
                print(f"\tDATA_LOADING_PERFORMANCE: reading time: {finish_reading - start_reading:.1f} resize time: {finish_resize - start_resize:.1f} rotate time: {finish_rotate - start_rotate:.1f} flip time: {finish_flip - start_flip:.1f}")

            yield tf.convert_to_tensor(data, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int8)

def craft_datasets(src_folder):
    result = None

    if os.path.isdir(src_folder):
        utils = ds_generator_factory(config)
        read_and_resize = Array3d_read_and_resize(src_folder)

        list_ds = tf.data.Dataset\
                    .from_generator(
                        read_and_resize, 
                        args=[], 
                        output_signature=(
                            tf.TensorSpec(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z], dtype = tf.float32),
                            tf.TensorSpec(shape = utils.ds_label_shape(), dtype = tf.int32)
                        ),
                    )\
                    .map(utils.expand_dimension)\
                    .batch(config.BATCH_SIZE)\
                    # .prefetch(1)

        result = list_ds
    else:
        print("can't craft dataset, folder {} doesn't exists".format(src_folder))

    return result




