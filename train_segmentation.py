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


def __dice_coef(y_true, y_pred):
    gamma = 0.01
    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred[..., 1:2], dtype = tf.float32)

    # print("y_true shape: ", y_true.shape)
    # print("y_pred shape: ", y_pred.shape)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + gamma) / (union + gamma)
    return dice

def __dice_loss(y_true, y_pred):
    return 1 - __dice_coef(y_true, y_pred)


def __weighted_loss(y_true, y_pred):
    scaler = 1

    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred, dtype = tf.float32)

    # count_0 = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    # count_1 = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))
    # background_weight = (1 - count_0 / (count_0 + count_1)) * config.LOSS_SCALER
    # foreground_weight = (1 - count_1 / (count_0 + count_1)) * config.LOSS_SCALER / 5

    # background_weight = config.BACKGROUND_WEIGHT  # incorrect loss calculations
    # foreground_weight = config.FOREGROUND_WEIGHT  # incorrect loss calculations
    # foreground_weight -= background_weight  # incorrect loss calculations

    # loss function described in the paper https://arxiv.org/pdf/1803.05431v2
    foreground_size = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))
    background_size = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    size = tf.cast(tf.size(y_true), dtype = tf.float32)
    foreground_weight = (1.0 - foreground_size / size) * scaler
    background_weight = (1.0 - background_size / size)
    # foreground and background weights must add up to 1 
    # it will be added/reversed later
    foreground_weight -= background_weight

    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    loss = scce(
        tf.maximum(y_true, 0.0),  # remove -1 values from mask,
        y_pred,
        sample_weight = tf.maximum(y_true * foreground_weight + background_weight, 0.0),
    )

    return loss

def __loss_func(loss_name):
    if loss_name == "dice":
        return __dice_loss
    elif loss_name == "scce":
        return __weighted_loss
    else:
        raise ValueError("Unknown loss function")

def main():
    ds_train = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"))
    ds_valid = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"))

    ds_train = ds_train.prefetch(1).repeat(config.TRAIN_PASSES_PER_VALIDATION)

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
    model.compile(optimizer = tf.keras.optimizers.AdamW(
                                                        learning_rate = config.INITIAL_LEARNING_RATE,
                                                        # gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
                                                        ),
                  loss = __loss_func(config.LOSS_FUNCTION),
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
            # tf.keras.callbacks.TerminateOnNaN()
            ],
        verbose = 1,
        # workers = 2
    )


def test_loss():
    # loss_fn = __dice_loss
    loss_fn = __weighted_loss

    y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
    y_pred = np.array([[[[[0.1, 4.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    print("shape of y_true: ", y_true.shape)
    print("shape of y_pred: ", y_pred.shape)

    loss = loss_fn(y_true, y_pred)
    print("expect 1, predict 1: loss {:.5f}".format(loss.numpy()))

    y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
    y_pred = np.array([[[[[0.1, 0.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

    y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
    y_pred = np.array([[[[[4.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect 1, predict 0: loss {:.4f}".format(loss.numpy()))

    y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
    y_pred = np.array([[[[[0.1, 0.9], [0.1, 4.9]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect 0, predict 1: loss {:.4f}".format(loss.numpy()))

    print("--- big volume")

    mx_size = 200
    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_true[0, 10:11, 10:11, 10:11, 0] = 1
    y_pred[0, 10:11, 10:11, 10:11, 1] = 1
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect\t\t1, predict\t1: loss {:.4f}".format(loss.numpy()))

    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_true[0, 10:11, 10:11, 10:11, 0] = 1.0
    y_pred[0, 10:11, 10:11, 10:11, 1] = 0.5
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    loss = loss_fn(y_true, y_pred)
    print("expect\t\t1, predict    0.5: loss {:.4f}".format(loss.numpy()))

    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_true[0, 10:11, 10:11, 10:11, 0] = 1
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect\t\t1, predict\t0: loss {:.4f}".format(loss.numpy()))

    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_true[0, 10:12, 10:11, 10:11, 0] = 1
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect\t    2 x 1, predict\t0: loss {:.4f}".format(loss.numpy()))

    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_pred[0, 10:11, 10:11, 10:11, 1] = 1
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect\t\t0, predict\t1: loss {:.4f}".format(loss.numpy()))

    y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
    y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
    y_pred[0, 10:12, 10:11, 10:11, 1] = 1
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    loss = loss_fn(y_true, y_pred)
    print("expect\t\t0, predict  2 x 1: loss {:.4f}".format(loss.numpy()))


if __name__ == "__main__":
    main()
    # test_loss()
