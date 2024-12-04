import os
import tensorflow as tf
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai.dataset.craft_datasets import craft_datasets
from pancreas_ai.tools.craft_network import factory
from pancreas_ai.tools.craft_network.loss import loss_func_factory
from pancreas_ai.tools.craft_network.metrics import metrics_factory
from pancreas_ai import config


def get_tensorboard_log_dir():
    root_log_dir = os.path.join(os.curdir, "logs")
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def get_csv_dir():
    root_log_dir = os.path.join(os.curdir, "csv_logs")
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def main():
    ds_valid = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "valid"), config)
    ds_train = craft_datasets(os.path.join(config.TFRECORD_FOLDER, "train"), config)

    ds_train = ds_train.prefetch(1).repeat(config.TRAIN_PASSES_PER_VALIDATION)

    model = factory.model_factory(config)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                                                    config.MODEL_CHECKPOINT,
                                                    monitor = config.MONITOR_METRIC,
                                                    mode = config.MONITOR_MODE,
                                                    verbose = 2,
                                                    # save_best_only = True
                                                    )
    csv_logger = tf.keras.callbacks.CSVLogger(get_csv_dir(), separator = ',', append = True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_tensorboard_log_dir())
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor = 0.5,
                                                                monitor = config.MONITOR_METRIC,
                                                                patience = 30,
                                                                cooldown = 10,
                                                                min_lr = 1e-6,
                                                                verbose = 1,
                                                                mode = config.MONITOR_MODE)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = config.MONITOR_METRIC, mode = config.MONITOR_MODE, patience = 200,
                                                      verbose = 1)
    model.compile(
        optimizer = tf.keras.optimizers.AdamW(
                                                        learning_rate = config.INITIAL_LEARNING_RATE,
                                                        # gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
                                                        ),
        loss = loss_func_factory(config),
        metrics = metrics_factory(config),
    )

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

if __name__ == "__main__":
    main()
