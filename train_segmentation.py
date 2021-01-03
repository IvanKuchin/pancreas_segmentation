import tensorflow as tf
import numpy as np
import time
import os
from tools.craft_network import craft_network

from dataset.craft_datasets import craft_datasets, py_read_data_and_label, crop_to_shape
from tools.craft_network import craft_network
from tools.categorical_metrics import CategoricalMetric, CategoricalF1


TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"


def run_through_data_wo_any_action(ds_train, ds_valid):
    print("Train ds:")
    for idx, (data, label) in enumerate(ds_train):
        print("---", idx)
        print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
        print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype = tf.float32)))
        # print(label)
        # break

    print("Valid ds:")
    for idx, (data, label) in enumerate(ds_valid):
        print("---", idx)
        print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
        print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype = tf.float32)))


def predict_on_random_data(model):
    for i in range(1, 32):
        t0 = time.time()
        pred = model.predict(tf.random.normal([1, 256, 256, 256, 1]))
        t1 = time.time()
        print("prediction shape/time:", pred.shape, "/", np.round(t1 - t0))


def get_tensorboard_log_dir():
    root_log_dir = os.path.join(os.curdir, "logs")
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def __custom_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    loss = scce(y_true, y_pred, sample_weight = y_true*5000+1)
    return loss


def main():
    ds_train, ds_valid = craft_datasets(TFRECORD_FOLDER)
    # run_through_data_wo_any_action(ds_train, ds_valid)

    model = craft_network("weights.hdf5")
    # predict_on_random_data(model)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("pancreas_segmentation_checkpoint.h5")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_tensorboard_log_dir())

    model.compile(optimizer = "adam", loss = __custom_loss,
                  metrics = [
                      'accuracy',
                      CategoricalMetric(tf.keras.metrics.TruePositives(), name = 'custom_tp'),
                      CategoricalMetric(tf.keras.metrics.FalsePositives(), name = 'custom_fp'),
                      CategoricalMetric(tf.keras.metrics.TrueNegatives(), name = 'custom_tn'),
                      CategoricalMetric(tf.keras.metrics.FalseNegatives(), name = 'custom_fn'),
                      CategoricalMetric(tf.keras.metrics.Accuracy(), name = 'custom_accuracy'),
                      CategoricalMetric(tf.keras.metrics.Precision(), name = 'custom_precision'),
                      CategoricalMetric(tf.keras.metrics.Recall(), name = 'custom_recall'),
                      CategoricalF1(name = 'custom_f1'),
                  ])

    history = model.fit(ds_train, epochs = 1, validation_data = ds_valid, callbacks = [checkpoint_cb, tensorboard_cb])

    # model.save("pancreas_segmentation_model.h5")


def main1():

    arr1, arr2 = py_read_data_and_label(tf.constant("c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_data.npy"),
                                        tf.constant("c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_label.npy"))
    print(arr1.shape)
    print(arr2.shape)


def main2():
    arr1 = tf.random.uniform([2, 270, 270, 270])
    random_offset = crop_to_shape(arr1, arr1)
    print(tf.shape(random_offset[0]))


if __name__ == "__main__":
    main()
    # main1()
    # main2()
