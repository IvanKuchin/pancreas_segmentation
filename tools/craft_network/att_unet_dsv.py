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
from tools.craft_network.dsv import DSV

import tools.config as config


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

    dsv_outputs = []
    skip_conns = reversed(generator_steps_output[:-1])
    for idx, (_filter, skip_conn) in enumerate(zip(reversed(filters[:-1]), skip_conns)):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = [4,4,1], strides = [2,2,1], padding = "same", kernel_initializer='he_uniform')(x)

        if _filter == filters[0]:
            # --- don't gate signal due to no useful features at top level
            gated_skip = skip_conn
        else:
            gated_skip = AttGate(apply_batchnorm = apply_batchnorm)((skip_conn, gating_base))

        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, gated_skip])
        x = double_conv(_filter, kernel_size = config.KERNEL_SIZE, apply_batchnorm = apply_batchnorm)(x)

        dsv_outputs.append(DSV(filters[-1] // filters[idx + 1])(x))
        # print("{} -> {}".format(x.shape, dsv_outputs[-1].shape))

    concat_layer = tf.keras.layers.Concatenate()(dsv_outputs)
    # print("concat layer shape: {}".format(concat_layer.shape))

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform")(concat_layer)
    # output_layer = tf.keras.activations.softmax(output_layer)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [output_layer])

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file {} not found".format(checkpoint_file))

    return model


def layer_info(layer):
    theta = layer.theta
    weights = theta.get_weights()
    print("Layer name: ", layer.name)
    # print("Layer input shape: ", layer.input_shape)
    # print("Layer output shape: ", layer.output_shape)
    print("Layer config: ", layer.get_config())


def main():
    import numpy as np 
    
    model_original = craft_network("", apply_batchnorm = True)
    model_original.summary(line_length = 128)
    model_original.save("test.keras")
    model_original.save_weights("test.weights.h5")

    y_original = model_original(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
    print("Model output shape: ", y_original.shape)

    model_reconstructed = craft_network("test.keras", apply_batchnorm = True)
    y_reconstructed = model_reconstructed(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
    print("diff after crafting the model: ", np.sum(y_reconstructed - y_original))
    model_reconstructed.load_weights("test.weights.h5")
    y_reconstructed = model_reconstructed(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
    print("diff after load_weights: ", np.sum(y_reconstructed - y_original))


    # model_reconstructed = craft_network("test.weights.h5", apply_batchnorm = False)
    # y_reconstructed = model_reconstructed(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
    # print("diff after load_weights: ", np.sum(y_reconstructed - y_original))
    
    # model.load_weights("test.weights.h5")
    # model.save("custom.keras")
    # check = tf.keras.models.load_model("custom.keras")


    # predict_on_random_data(model)


if __name__ == "__main__":
    main()
