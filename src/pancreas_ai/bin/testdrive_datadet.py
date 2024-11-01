import os
import sys
import numpy as np
import time
import tensorflow as tf


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai.dataset.craft_datasets import craft_datasets
import pancreas_ai.config as config

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

    ds_train = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"))
    ds_valid = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"))
    __run_through_data_wo_any_action(ds_train, ds_valid)