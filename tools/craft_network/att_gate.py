import tensorflow as tf

def attention_gate(x, gated, apply_batchnorm=True):
    assert x.shape[0] == gated.shape[0]
    assert x.shape[1] >= gated.shape[1]
    assert x.shape[2] >= gated.shape[2]
    assert x.shape[3] >= gated.shape[3]

    # input_x = tf.keras.layers.Input(shape = x.shape[1:])
    # input_g = tf.keras.layers.Input(shape = g.shape[1:])

    __inter_filters = x.shape[-1]

    phi_g = tf.keras.layers.Conv3D(__inter_filters, kernel_size = 1, strides = 1, padding = "same", kernel_initializer="he_uniform")(gated)
    theta_x = tf.keras.layers.Conv3D(__inter_filters, kernel_size = 1, strides = 1, padding = "same", kernel_initializer="he_uniform")(x)

    # theta_x_shape = theta_x.shape[1:-1]
    # phi_g_shape = phi_g.shape[1:-1]
    #
    # phi_g_upsampled = tf.keras.layers.UpSampling3D(tf.divide(theta_x_shape, phi_g_shape))(phi_g)
    phi_g_upsampled = tf.keras.layers.UpSampling3D([
        theta_x.shape[1] // phi_g.shape[1],
        theta_x.shape[2] // phi_g.shape[2],
        theta_x.shape[3] // phi_g.shape[3]
    ])(phi_g)

    __sum = tf.keras.layers.Add()([phi_g_upsampled, theta_x])

    __activation_sum = tf.keras.layers.Activation("relu")(__sum)

    psi = tf.keras.layers.Conv3D(filters = 1, kernel_size = 1, strides = 1, padding = "same", kernel_initializer="he_uniform")(__activation_sum)

    __activation_psi = tf.keras.layers.Activation("sigmoid")(psi)

    __mul = tf.keras.layers.Multiply()([x, __activation_psi])

    # print("phi_g {}, phi_g_upsampled {}, theta_x {}, sum {}".format(phi_g.shape, phi_g_upsampled.shape, theta_x.shape, __sum.shape))
    # print("psi {}, mul {}".format(psi.shape, __mul.shape))

    __result = tf.keras.layers.Conv3D(filters = x.shape[-1], kernel_size = 1, strides = 1, padding = "same", kernel_initializer="he_uniform")(__mul)
    if apply_batchnorm:
        __result = tf.keras.layers.BatchNormalization()(__result)

    # return tf.keras.models.Model(inputs = [input_x, input_g], outputs = [__result])
    return __result


if __name__ == "__main__":
    rnd = tf.random.uniform([3, 64, 64, 256, 16])

    inp = tf.keras.layers.Input(shape = rnd.shape[1:])
    x = tf.keras.layers.Conv3D(16, kernel_size = 1, strides = 1, padding = "same")(inp)
    gated = tf.keras.layers.Conv3D(128, kernel_size = (8, 8, 1), strides = (4, 4, 1), padding = "same")(inp)

    result = attention_gate(x = x, gated = gated, apply_batchnorm = False)
    model = tf.keras.models.Model(inputs = [inp], outputs = [result, x, gated])
    outputs = model(rnd)

    print("x {}, g {}, result {}".format(outputs[1].shape, outputs[2].shape, outputs[0].shape))