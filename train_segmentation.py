import tensorflow as tf
import numpy as np
import time
import os

from dataset.craft_datasets import craft_datasets, py_read_data_and_label, crop_to_shape
from tools.categorical_metrics import CategoricalMetric, CategoricalF1
from tools.craft_network import craft_network
import tools.config as config


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

    count_0 = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    count_1 = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))

    background_weight = (1 - count_0 / (count_0 + count_1)) * config.LOSS_SCALER
    foreground_weight = (1 - count_1 / (count_0 + count_1)) * config.LOSS_SCALER / 5

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
    ds_train, ds_valid = craft_datasets(config.TFRECORD_FOLDER)
    # run_through_data_wo_any_action(ds_train, ds_valid)

    model = craft_network(config.MODEL_CHECKPOINT)
    # predict_on_random_data(model)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(config.MODEL_CHECKPOINT, monitor = config.MONITOR_METRIC,
                                                       mode = config.MONITOR_MODE, verboose = 2, save_best_only = True)
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
        workers = 2
    )


if __name__ == "__main__":
    main()
