import tensorflow as tf
from dataset.craft_datasets import craft_datasets
from tools.craft_network.unet_shortcuts_every_layer import craft_network
from tools.categorical_metrics import CategoricalMetric, CategoricalF1
import tools.config as config

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
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    loss = scce(
        y_true,
        y_pred,
        sample_weight = y_true * config.WEIGHT_SCALE + config.WEIGHT_BIAS
        )
    return loss


def main():
    ds_train, ds_valid = craft_datasets(config.TFRECORD_FOLDER)
    data_orig, label_orig = next(iter(ds_valid))
    data_orig = data_orig[0:1, ...]
    label_orig = label_orig[0:1, ...]

    # model = tf.keras.models.load_model("../pancreas_segmentation_model.h5", compile=False)
    model = craft_network("../weights.hdf5")
    # latest = tf.train.latest_checkpoint("./", "pancreas_segmentation_checkpoint.h5")
    # print(latest)
    # model.load_weights("/Users/ikuchin/PycharmProjects/ct_prediction/pancreas_segmentation_checkpoint.h5")
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

    model.evaluate_generator(ds_train, use_multiprocessing = True, verbose=1)

    for data, label in ds_valid:
        model.evaluate(data, label)

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
