import tensorflow as tf

def attention_gate(x, g, apply_batchnorm=True):
    assert x.shape[0] == g.shape[0]
    assert x.shape[1] == g.shape[1]
    assert x.shape[2] == g.shape[2]
    assert x.shape[3] == g.shape[3]


    x_reshaped = x

    if x.shape[-1] != g.shape[-1]:
        x_reshaped = tf.keras.layers.Conv3D(filters = g.shape[-1], kernel_size = 1, padding = "same")(x)

    inter_filters = g.shape[-1] // 2

    phi_g = tf.keras.layers.Conv3D(inter_filters, kernel_size = 1, strides = 1, padding = "same")(g)
    theta_x = tf.keras.layers.Conv3D(inter_filters, kernel_size = 1, strides = 1, padding = "same")(x_reshaped)

    __sum = tf.keras.layers.Add()([phi_g, theta_x])

    __activation_sum = tf.keras.layers.Activation("relu")(__sum)

    psi = tf.keras.layers.Conv3D(filters = 1, kernel_size = 1, strides = 1, padding = "same")(__activation_sum)

    __activation_psi = tf.keras.layers.Activation("sigmoid")(psi)

    __mul = tf.keras.layers.Multiply()([x, __activation_psi])

    __result = tf.keras.layers.Conv3D(filters = x.shape[-1], kernel_size = 1, strides = 1, padding = "same")(__mul)
    if apply_batchnorm:
        __result = tf.keras.layers.BatchNormalization()(__result)

    return __result


if __name__ == "__main__":
    x = tf.random.uniform([3, 64, 64, 64, 16])
    gate = tf.random.uniform([3, 64, 64, 64, 32])

    result = attention_gate(x, gate, apply_batchnorm = False)

    print(result.shape)
