import tensorflow as tf
import os

import tools.config as config
from tools.craft_network.att_gate import attention_gate


def model_step(filters, kernel_size=[3,3,3], apply_batchnorm=True, apply_dropout=False):
    model = tf.keras.models.Sequential()

    if apply_batchnorm:
        model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
        model.add(tf.keras.layers.ReLU())
    else:
        model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform', activation = "relu"))

    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))

    return model


def craft_network(checkpoint_file = None, apply_batchnorm=True):
    filters = [32, 64, 128, 256, 512]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs

    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        x = model_step(_filter, apply_batchnorm=apply_batchnorm)(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.Conv3D(_filter, kernel_size = 3, strides=2, padding = "same", kernel_initializer="he_uniform", activation = "relu")(x)

    skip_conns = reversed(generator_steps_output[:-1])
    for _filter, skip_conn in zip(reversed(filters[:-1]), skip_conns):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = 3, strides = 2, padding = "same", kernel_initializer='he_uniform', activation = "relu")(x)

        gated_skip_conn = attention_gate(x = skip_conn, gated = x, apply_batchnorm = apply_batchnorm)

        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, gated_skip_conn])
        x = model_step(_filter, apply_batchnorm=apply_batchnorm)(x)

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform")(x)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [output_layer])

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file {} not found".format(checkpoint_file))

    return model

