import tensorflow as tf
import numpy as np
import glob
import os
from craft_datasets import craft_datasets, py_read_data_and_label, crop_to_shape


TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"


def get_min_mean_max(ds):
    for data, labels in ds:
        print("data, labels: {}/{:.2f}/{}, {}/{:.4f}/{}".format(
            tf.reduce_min(data).numpy(),
            tf.reduce_mean(data).numpy(),
            tf.reduce_max(data).numpy(),
            tf.reduce_min(labels).numpy(),
            tf.reduce_mean(tf.cast(labels, dtype=tf.float32)).numpy(),
            tf.reduce_max(labels).numpy(),
        ))

        if(data.shape != labels.shape):
            print_error("shapea are not equal {} and {}".format(data.shape, labels.shape))

def main():
    train_ds, valid_ds = craft_datasets(TFRECORD_FOLDER)
    print("--- Train set")
    get_min_mean_max(train_ds)
    print("--- Valid set")
    get_min_mean_max(valid_ds)

def print_file_stat(file_name):
    arr = np.load(file_name)
    print("{}: {}/{}/{}".format(file_name, np.min(arr), np.mean(arr), np.max(arr)))

def main2():
    for file_name in glob.glob(os.path.join(TFRECORD_FOLDER, "*_data.npy")):
        print_file_stat(file_name)

if __name__ == "__main__":
    main()
    # main2()