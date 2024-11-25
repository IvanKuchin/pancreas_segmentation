import tensorflow as tf
import os

from .att_gate import AttGate

def res_block(filters, input_shape, config: dict):
    input_layer = tf.keras.layers.Input(shape = input_shape[1:])

    # primary path
    x = tf.keras.layers.Conv3D(filters, kernel_size = config.KERNEL_SIZE, padding = "same", kernel_initializer='he_uniform',  kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    if (config.BATCH_NORM):
        x = tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    if (config.DROPOUT):
        x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv3D(filters, kernel_size = config.KERNEL_SIZE, padding = "same", kernel_initializer='he_uniform',  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    if (config.BATCH_NORM):
        x = tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM)(x)

    # residual path
    res_path = input_layer
    if (input_shape[-1] != filters):
        res_path = tf.keras.layers.Conv3D(filters, kernel_size = 1, padding = "same", kernel_initializer='he_uniform',  kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
        if (config.BATCH_NORM):
            res_path = tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM)(res_path)

    # add the residual path
    x = tf.keras.layers.Add()([res_path, x])

    x = tf.keras.layers.LeakyReLU()(x)
    if (config.DROPOUT):
        x = tf.keras.layers.Dropout(0.5)(x)
    
    model =  tf.keras.models.Model(inputs = input_layer, outputs = x, name = "res_block_{}_{}".format(input_shape[-1], filters))
    return model

def double_conv(filters, input_shape, kernel_size, config: dict):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (config.BATCH_NORM):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    if (config.INSTANCE_NORM):
        # Instance normalization is not supported in tf.keras.layers.BatchNormalization
        model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if (config.DROPOUT):
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv3D(filters, kernel_size = kernel_size, padding = "same", kernel_initializer='he_uniform'))
    if (config.BATCH_NORM):
        model.add(tf.keras.layers.BatchNormalization(momentum=config.BATCH_NORM_MOMENTUM))
    if (config.INSTANCE_NORM):
        # Instance normalization is not supported in tf.keras.layers.BatchNormalization
        model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if (config.DROPOUT):
        model.add(tf.keras.layers.Dropout(0.5))

    return model

def get_gating_base(filters, config: dict):
    __model = tf.keras.models.Sequential(name = "gating_base")
    __model.add(tf.keras.layers.Conv3D(filters, kernel_size = 1, padding = "same", kernel_initializer='he_uniform'))

    return __model

def craft_network(config: dict):
    checkpoint_file = config.MODEL_CHECKPOINT
    apply_batchnorm = config.BATCH_NORM
    apply_instancenorm = config.INSTANCE_NORM
    
    filters = [16, 32, 64, 128, 256]

    inputs = tf.keras.layers.Input(shape = [config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1])

    x = inputs
    generator_steps_output = []
    for idx, _filter in enumerate(filters):
        resBlock = res_block(_filter, x.shape, config)
        x = resBlock(x)
        generator_steps_output.append(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding = "same")(x)

    gating_base = get_gating_base(filters[-2], config)(x)

    skip_conns = reversed(generator_steps_output[:-1])
    for _filter, skip_conn in zip(reversed(filters[:-1]), skip_conns):
        x = tf.keras.layers.Conv3DTranspose(_filter, kernel_size = [4,4,4], strides = [2,2,2], padding = "same", kernel_initializer='he_uniform')(x)

        if _filter == filters[0]:
            # --- don't gate signal due to no useful features at top level
            gated_skip = skip_conn
        else:
            # gated_skip = attention_gate(skip_conn, gating_base, apply_batchnorm = apply_batchnorm)
            gated_skip = AttGate(apply_batchnorm = apply_batchnorm)((skip_conn, gating_base))

        x = tf.keras.layers.Concatenate(name = "concat_{}".format(_filter))([x, gated_skip])
        x = res_block(_filter, x.shape, config)(x)

    output_layer = tf.keras.layers.Conv3D(2, kernel_size = 1, padding = "same", kernel_initializer = "he_uniform", activation="softmax")(x)

    model = tf.keras.models.Model(inputs = [inputs], outputs = output_layer)

    if checkpoint_file and os.path.exists(checkpoint_file):
        print("Loading weights from checkpoint ", checkpoint_file)
        model(tf.ones(shape=(1, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1)))
        model.load_weights(checkpoint_file)
    else:
        print("Checkpoint file {} not found".format(checkpoint_file))

    return model

