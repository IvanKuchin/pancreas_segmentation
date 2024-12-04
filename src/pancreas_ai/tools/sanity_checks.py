import tensorflow as tf
import numpy as np
import glob
import os
from dataset.craft_datasets import craft_datasets

config.TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"


def get_min_mean_max(ds):
    print("data, labels: min/mean/max, min/mean/max --- sum/shape/percentage")
    for idx, (data, labels) in enumerate(ds):
        number_of_items = data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]

        print("{}) data, labels: {}/{:.2f}/{}, {}/{:.4f}/{} --- {}/{}/{:5.2f}%".format(
            idx,
            tf.reduce_min(data).numpy(),
            tf.reduce_mean(data).numpy(),
            tf.reduce_max(data).numpy(),
            tf.reduce_min(labels).numpy(),
            tf.reduce_mean(tf.cast(labels, dtype=tf.float32)).numpy(),
            tf.reduce_max(labels).numpy(),
            tf.reduce_sum(labels).numpy(),
            number_of_items,
            tf.reduce_sum(labels).numpy() / number_of_items * 100
        ))

        if(data.shape != labels.shape):
            print_error("shapea are not equal {} and {}".format(data.shape, labels.shape))

def main():
    train_ds, valid_ds = craft_datasets(config.TFRECORD_FOLDER, config)
    print("--- Train set")
    get_min_mean_max(train_ds)
    print("--- Valid set")
    get_min_mean_max(valid_ds)

def print_file_stat(file_name):
    arr = np.load(file_name)
    print("{}: {}/{}/{}".format(file_name, np.min(arr), np.mean(arr), np.max(arr)))

def main2():
    for file_name in glob.glob(os.path.join(config.TFRECORD_FOLDER, "*_data.npy")):
        print_file_stat(file_name)

if __name__ == "__main__":
    main()
    # main2()