import tensorflow as tf
import numpy as np
from dataset.craft_datasets import craft_datasets
from tools.craft_network import craft_network

TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"


def __print_stat(data, title=""):
    print('-' * 100)
    if len(title):
        print(title)
    print("shape", data.shape)
    print("min/mean/max {}/{:.2}/{}".format(tf.reduce_min(data), tf.reduce_mean(tf.cast(data, dtype=tf.float32)), tf.reduce_max(data)))

def __create_mask(data):
    mask = tf.argmax(data, axis = -1)
    mask = mask[..., tf.newaxis]
    return mask


def __custom_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    loss = scce(y_true, y_pred, sample_weight = y_true*5000+1)
    return loss


def main():
    ds_train, ds_valid = craft_datasets(TFRECORD_FOLDER)
    data_orig, label_orig = next(iter(ds_valid))
    data_orig = data_orig[0:1, ...]
    label_orig = label_orig[0:1, ...]

    # model = tf.keras.models.load_model("../pancreas_segmentation_model.h5", compile=False)
    model = craft_network()
    # latest = tf.train.latest_checkpoint("./", "pancreas_segmentation_checkpoint.h5")
    # print(latest)
    model.load_weights("/Users/ikuchin/PycharmProjects/ct_prediction/pancreas_segmentation_checkpoint.h5")
    model.compile(optimizer = "adam", loss = __custom_loss,
                  metrics = ['accuracy', 'sparse_categorical_crossentropy'])
    model.evaluate(data_orig, label_orig)

    # label_pred = model.predict(data_orig)
    # mask = __create_mask(label_pred)
    #
    # __print_stat(label_orig, "label orig")
    # __print_stat(mask, "mask (must be as close as possible to orig)")
    # if tf.reduce_mean(tf.cast(mask, tf.float32)) == 0:
    #     __print_stat(label_pred, "label pred")
    #     __print_stat(label_pred[:, :, :, :, 0], "label pred[0]")
    #     __print_stat(label_pred[:, :, :, :, 1], "label pred[1]")
    #
    # accuracy = tf.keras.metrics.Accuracy()
    # accuracy.update_state(label_orig, mask)
    # print("-" * 100)
    # print("accuracy:", accuracy.result())

if __name__ == "__main__":
    main()
