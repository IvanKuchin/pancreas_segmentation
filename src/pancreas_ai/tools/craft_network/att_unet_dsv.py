import tensorflow as tf
import os

from .att_gate import AttGate
from .dsv import DSV
from .att_unet import res_block, get_gating_base

def craft_network(config: dict):
    checkpoint_file = config.MODEL_CHECKPOINT
    apply_batchnorm = config.BATCH_NORM

    filters = [16, 32, 64, 128, 256]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs
    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        x = res_block(_filter, x.shape, config)(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding = "same")(x)

    gating_base = get_gating_base(filters[-1], config)(x)

    dsv_outputs = []
    skip_conns = reversed(generator_steps_output[:-1])
    for idx, (_filter, skip_conn) in enumerate(zip(reversed(filters[:-1]), skip_conns)):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = [4,4,4], strides = [2,2,2], padding = "same", kernel_initializer='he_uniform')(x)

        if _filter == filters[0]:
            # --- don't gate signal due to no useful features at top level
            gated_skip = skip_conn
        else:
            if idx == 0:
                gated_skip = AttGate(apply_batchnorm = apply_batchnorm)((skip_conn, gating_base))
            gated_skip = AttGate(apply_batchnorm = apply_batchnorm)((skip_conn, x))

        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, gated_skip])
        x = res_block(_filter, x.shape, config)(x)

        dsv_outputs.append(DSV(filters[-1] // filters[idx + 1])(x))
        # print("{} -> {}".format(x.shape, dsv_outputs[-1].shape))

    concat_layer = tf.keras.layers.Concatenate()(dsv_outputs)
    # print("concat layer shape: {}".format(concat_layer.shape))

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform", activation="softmax")(concat_layer)
    # output_layer = tf.keras.activations.softmax(output_layer)

    model = tf.keras.models.Model(inputs = inputs, outputs = output_layer)

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file {} not found".format(checkpoint_file))

    return model

