import tensorflow as tf
import os

def double_conv(filters, config: dict):
    """
    Sequential model with two convolutional layers, each followed by a batch normalization layer, a ReLU activation and a dropout layer.
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = config.KERNEL_SIZE, padding = "same", kernel_initializer='he_uniform'))
    if (config.BATCH_NORM):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    model.add(tf.keras.layers.ReLU())
    if (config.DROPOUT):
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = config.KERNEL_SIZE, padding = "same", kernel_initializer='he_uniform'))
    if (config.BATCH_NORM):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    model.add(tf.keras.layers.ReLU())
    if (config.DROPOUT):
        model.add(tf.keras.layers.Dropout(0.5))

    return model


def craft_network(config: dict):
    filters = [32, 64, 128, 256]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs
    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        x = double_conv(_filter, config)(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding = "same")(x)

    skip_conns = reversed(generator_steps_output[:-1])
    for _filter, skip_conn in zip(reversed(filters[:-1]), skip_conns):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = 3, strides = 2, padding = "same", kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, skip_conn])
        x = double_conv(_filter, config)(x)

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform")(x)

    model = tf.keras.models.Model(inputs = [inputs], outputs = output_layer)

    if config.MODEL_CHECKPOINT and os.path.exists(config.MODEL_CHECKPOINT):
        print("Loading weights from checkpoint ", config.MODEL_CHECKPOINT)
        model.load_weights(config.MODEL_CHECKPOINT)
    else:
        print("Checkpoint file {} not found".format(config.MODEL_CHECKPOINT))

    return model
