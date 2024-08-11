import tensorflow as tf
import os
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

from tools.predict_on_random_data import predict_on_random_data
from tools.craft_network.att_gate import AttGate

import config as config


def double_conv(filters, kernel_size=[3,3,1], apply_batchnorm=True, apply_dropout=False):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (apply_batchnorm):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    model.add(tf.keras.layers.ReLU())
    if (apply_dropout):
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (apply_batchnorm):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    model.add(tf.keras.layers.ReLU())
    if (apply_dropout):
        model.add(tf.keras.layers.Dropout(0.5))

    return model

def get_gating_base(filters, apply_batchnorm = True):
    __model = tf.keras.models.Sequential(name = "gating_base")

    __model.add(tf.keras.layers.Conv3D(filters, kernel_size = 1, padding = "same", kernel_initializer='he_uniform'))
    if apply_batchnorm:
      __model.add(tf.keras.layers.BatchNormalization(momentum = config.BATCH_NORM_MOMENTUM))

    return __model


def craft_network(checkpoint_file = None, apply_batchnorm=True):
    filters = [16, 32, 64, 128, 256]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs
    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        x = double_conv(_filter, kernel_size = config.KERNEL_SIZE, apply_batchnorm = apply_batchnorm)(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), padding = "same")(x)

    gating_base = get_gating_base(filters[-2], apply_batchnorm)(x)

    skip_conns = reversed(generator_steps_output[:-1])
    for _filter, skip_conn in zip(reversed(filters[:-1]), skip_conns):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = [4,4,1], strides = [2,2,1], padding = "same", kernel_initializer='he_uniform')(x)

        if _filter == filters[0]:
            # --- don't gate signal due to no useful features at top level
            gated_skip = skip_conn
        else:
            # gated_skip = attention_gate(skip_conn, gating_base, apply_batchnorm = apply_batchnorm)
            gated_skip = AttGate(apply_batchnorm = apply_batchnorm)((skip_conn, gating_base))

        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, gated_skip])
        x = double_conv(_filter, kernel_size = config.KERNEL_SIZE, apply_batchnorm = apply_batchnorm)(x)

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform", activation="softmax")(x)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [output_layer])

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file {} not found".format(checkpoint_file))

    return model


def main():
    model = craft_network("checkpoints/weights.keras", apply_batchnorm = config.BATCH_NORM)

    model.summary(line_length = 128, expand_nested = True, show_trainable = True)
    predict_on_random_data(model)


if __name__ == "__main__":
    main()
