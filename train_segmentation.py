import tensorflow as tf
import numpy as np
import time
import os

from dataset.craft_datasets import craft_datasets, py_read_data_and_label, crop_to_shape
from tools.categorical_metrics import CategoricalMetric, CategoricalF1, CustomCounter, CustomReduceMetric
from tools.craft_network import craft_network
import config as config


def get_tensorboard_log_dir():
    root_log_dir = os.path.join(os.curdir, "logs")
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def get_csv_dir():
    root_log_dir = os.path.join(os.curdir, "csv_logs")
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def __custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred, dtype = tf.float32)

    # count_0 = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    # count_1 = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))

    # background_weight = (1 - count_0 / (count_0 + count_1)) * config.LOSS_SCALER
    # foreground_weight = (1 - count_1 / (count_0 + count_1)) * config.LOSS_SCALER / 5

    background_weight = config.BACKGROUND_WEIGHT
    foreground_weight = config.FOREGROUND_WEIGHT
    foreground_weight -= background_weight

    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    loss = scce(
        tf.maximum(y_true, 0.0),  # remove -1 values from mask,
        y_pred,
        sample_weight = tf.maximum(y_true * foreground_weight + background_weight, 0.0)
    )

    return loss


def main():
    ds_train = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"))
    ds_valid = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"))

    ds_train = ds_train.prefetch(3).repeat(config.TRAIN_PASSES_PER_VALIDATION)

    model = craft_network(config.MODEL_CHECKPOINT)
    # predict_on_random_data(model)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                                                    config.MODEL_CHECKPOINT, 
                                                    monitor = config.MONITOR_METRIC,
                                                    mode = config.MONITOR_MODE, 
                                                    verbose = 2, 
                                                    # save_best_only = True
                                                    )
    csv_logger = tf.keras.callbacks.CSVLogger(get_csv_dir(), separator = ',', append = True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_tensorboard_log_dir())
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1,
                                                                monitor = config.MONITOR_METRIC,
                                                                patience = 30,
                                                                cooldown = 10,
                                                                min_lr = 0.000001,
                                                                verbose = 1,
                                                                mode = config.MONITOR_MODE)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = config.MONITOR_METRIC, mode = config.MONITOR_MODE, patience = 200,
                                                      verbose = 1)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = config.INITIAL_LEARNING_RATE),
                  loss = __custom_loss,
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
                      CustomReduceMetric(what = "y_true", reduce = "max", name = 'custom_max_y_true'),
                      CustomReduceMetric(what = "y_pred", reduce = "max", name = 'custom_max_y_pred'),
                      CustomReduceMetric(what = "y_true", reduce = "min", name = 'custom_min_y_true'),
                      CustomReduceMetric(what = "y_pred", reduce = "min", name = 'custom_min_y_pred'),
                      CustomReduceMetric(what = "y_true", reduce = "sum", name = 'custom_sum_y_true'),
                      CustomReduceMetric(what = "y_pred", reduce = "sum", name = 'custom_sum_y_pred'),
                      CustomCounter(name = 'custom_counter'),
                  ])

    history = model.fit(
        ds_train,
        epochs = config.EPOCHS,
        validation_data = ds_valid,
        callbacks = [
            checkpoint_cb,
            tensorboard_cb,
            reduce_lr_on_plateau,
            csv_logger,
            # early_stopping,
            tf.keras.callbacks.TerminateOnNaN()],
        verbose = 1,
        # workers = 2
    )


def test_loss():
    y_true = np.array([[[[[  1], [  0]], [[  0], [  0]]], [[[  0], [  0]], [[  0], [  0]]]]])
    y_pred = np.array([[[[[0.1, 4.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    print("shape of y_true: ", y_true.shape)
    print("shape of y_pred: ", y_pred.shape)

    loss = __custom_loss(y_true, y_pred)
    print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

    y_true = np.array([[[[[  1], [  0]], [[  0], [  0]]], [[[  0], [  0]], [[  0], [  0]]]]])
    y_pred = np.array([[[[[0.1, 0.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = __custom_loss(y_true, y_pred)
    print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

    y_true = np.array([[[[[  1], [  0]], [[  0], [  0]]], [[[  0], [  0]], [[  0], [  0]]]]])
    y_pred = np.array([[[[[4.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = __custom_loss(y_true, y_pred)
    print("expect 1, predict 0: loss {:.4f}".format(loss.numpy()))

    y_true = np.array([[[[[  1], [  0]], [[  0], [  0]]], [[[  0], [  0]], [[  0], [  0]]]]])
    y_pred = np.array([[[[[0.1, 0.9], [0.1, 4.9]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = __custom_loss(y_true, y_pred)
    print("expect 0, predict 1: loss {:.4f}".format(loss.numpy()))


if __name__ == "__main__":
    main()
    # test_loss()
