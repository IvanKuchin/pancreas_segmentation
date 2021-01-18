import tensorflow as tf
import os

from tools.predict_on_random_data import predict_on_random_data

import tools.config as config


def double_conv(filters, kernel_size=[3,3,3], apply_batchnorm=True, apply_dropout=False):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (apply_batchnorm):
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    if (apply_dropout):
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (apply_batchnorm):
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    if (apply_dropout):
        model.add(tf.keras.layers.Dropout(0.5))

    return model


def craft_network(checkpoint_file = None):
    downsample_steps = [
        double_conv(32, 3),   # (?, 128, 128, __filters__)
        double_conv(64, 3),   # (?,  64,  64, __filters__)
        double_conv(128, 3),  # (?,  32,  32, __filters__)
        double_conv(256, 3),  # (?,  16,  16, __filters__)
    ]

    upsample_steps = [
        double_conv(128, 3),  # (?,  32,  32, __filters__)
        double_conv(64, 3),   # (?,  64,  64, __filters__)
        double_conv(32, 3),   # (?, 128, 128, __filters__)
    ]

    filters = [32, 64, 128, 256]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs
    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        x = double_conv(_filter)(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding = "same")(x)

    skip_conns = reversed(generator_steps_output[:-1])
    for _filter, skip_conn in zip(reversed(filters[:-1]), skip_conns):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = 3, strides = 2, padding = "same", kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, skip_conn])
        x = double_conv(_filter)(x)

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform")(x)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [output_layer])

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file not found")

    return model


def main():
    model = craft_network("")

    model.summary(line_length = 128)
    predict_on_random_data(model)


if __name__ == "__main__":
    main()
