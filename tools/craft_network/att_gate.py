import tensorflow as tf

def attention_gate(x, gated, apply_batchnorm=True):
    assert x.shape[0] == gated.shape[0]
    assert x.shape[1] == gated.shape[1]
    assert x.shape[2] == gated.shape[2]
    assert x.shape[3] == gated.shape[3]

    # input_x = tf.keras.layers.Input(shape = x.shape[1:])
    # input_g = tf.keras.layers.Input(shape = g.shape[1:])

    inter_filters = gated.shape[-1] // 2

    phi_g = tf.keras.layers.Conv3D(inter_filters, kernel_size = 1, strides = 1, padding = "same")(gated)
    theta_x = tf.keras.layers.Conv3D(inter_filters, kernel_size = 1, strides = 1, padding = "same")(x)

    __sum = tf.keras.layers.Add()([phi_g, theta_x])

    __activation_sum = tf.keras.layers.Activation("relu")(__sum)

    psi = tf.keras.layers.Conv3D(filters = 1, kernel_size = 1, strides = 1, padding = "same")(__activation_sum)

    __activation_psi = tf.keras.layers.Activation("sigmoid")(psi)

    __mul = tf.keras.layers.Multiply()([x, __activation_psi])

    __result = tf.keras.layers.Conv3D(filters = x.shape[-1], kernel_size = 1, strides = 1, padding = "same")(__mul)
    if apply_batchnorm:
        __result = tf.keras.layers.BatchNormalization()(__result)

    # return tf.keras.models.Model(inputs = [input_x, input_g], outputs = [__result])
    return __result


if __name__ == "__main__":
    x = tf.random.uniform([3, 64, 64, 64, 16])
    gate = tf.random.uniform([3, 64, 64, 64, 32])

    inp = tf.keras.layers.Input(shape = x.shape[1:])
    conv1 = tf.keras.layers.Conv3D(16, kernel_size = 1, strides = 1, padding = "same")(inp)
    conv2 = tf.keras.layers.Conv3D(64, kernel_size = 1, strides = 1, padding = "same")(inp)

    result = attention_gate(x = conv1, gated = conv2, apply_batchnorm = False)

    model = tf.keras.models.Model(inputs = [inp], outputs = [result])

    print(model(x).shape)
