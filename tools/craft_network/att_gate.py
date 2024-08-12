import tensorflow as tf


class AttGate(tf.keras.layers.Layer):
    def __init__(self, apply_batchnorm=True, **kwargs):
        super().__init__(**kwargs)
        self.apply_batchnorm = apply_batchnorm

    def build(self, inputs):
        x_shape, gated_shape = inputs

        # assert x_shape.shape[0] == gated_shape.shape[0]
        assert x_shape[1] >= gated_shape[1]
        assert x_shape[2] >= gated_shape[2]
        assert x_shape[3] >= gated_shape[3]

        __inter_filters = x_shape[-1] // 2

        if(x_shape[1] == gated_shape[1]):
            subsample_factor = [1,1,1]
        else:
            subsample_factor = [2,2,2]

        # print("{} {}".format())

        self.theta = tf.keras.layers.Conv3D(__inter_filters, kernel_size = subsample_factor, strides = subsample_factor, padding = "same",
                                            use_bias = False, kernel_initializer = "he_uniform", name = "att_gate_theta")

        self.phi = tf.keras.layers.Conv3D(__inter_filters, kernel_size = 1, strides = 1, padding = "same",
                                          kernel_initializer = "he_uniform", name = "att_gate_phi")

        self.phi_upsample = tf.keras.layers.UpSampling3D([
            x_shape[1] // subsample_factor[0] // gated_shape[1],
            x_shape[2] // subsample_factor[1] // gated_shape[2],
            x_shape[3] // subsample_factor[2] // gated_shape[3]
        ])

        self.add_g_x = tf.keras.layers.Add(name = "att_gate_addition")

        self.psi = tf.keras.layers.Conv3D(filters = 1, kernel_size = 1, strides = 1, padding = "same",
                                          kernel_initializer = "he_uniform", name = "att_gate_psi")

        self.psi_upsample = tf.keras.layers.UpSampling3D(subsample_factor)

        self.multiplication_to_att = tf.keras.layers.Multiply(name = "att_gate_multiplication")

        self.W = tf.keras.layers.Conv3D(filters = x_shape[-1], kernel_size = 1, strides = 1, padding = "same",
                                        kernel_initializer = "he_uniform", name = "att_gate_W")
        
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x, gated = inputs

        # print("x {}, g {}".format(x.shape, gated.shape))
        theta_x = self.theta(x)
        # print("theta_x {}".format(theta_x.shape))
        phi_g = self.phi(gated)
        # print("phi_g {}".format(phi_g.shape))
        phi_g = self.phi_upsample(phi_g)
        # print("upsampled phi_g {}".format(phi_g.shape))


        __sum = self.add_g_x([theta_x, phi_g])
        __activation_sum = tf.keras.layers.Activation("relu")(__sum)

        psi = tf.keras.layers.Activation("sigmoid")(self.psi(__activation_sum))
        psi = self.psi_upsample(psi)
        # print("psi {}".format(psi.shape))

        __mul = self.multiplication_to_att([x, psi])

        # print("phi_g {}, phi_g_upsampled {}, theta_x {}, sum {}".format(phi_g.shape, phi_g_upsampled.shape, theta_x.shape, __sum.shape))
        # print("psi {}, mul {}".format(psi.shape, __mul.shape))

        __result = self.W(__mul)

        if self.apply_batchnorm:
            __result = self.bn(__result)

        # return tf.keras.models.Model(inputs = [input_x, input_g], outputs = [__result])
        return __result

    ##########################
    # ATTENTION !!! 
    # having this function will cause the model to fail to load weights from file
    ##########################
    # def compute_output_shape(self, input_shape):
    #     return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "apply_batchnorm": self.apply_batchnorm}

def __attention_gate(x, gated, apply_batchnorm=True):
    with tf.name_scope("att_gate"):
        assert x.shape[0] == gated.shape[0]
        assert x.shape[1] >= gated.shape[1]
        assert x.shape[2] >= gated.shape[2]
        assert x.shape[3] >= gated.shape[3]

        __filters = x.shape[-1]

        # input_x = tf.keras.layers.Input(shape = x.shape[1:])
        # input_g = tf.keras.layers.Input(shape = g.shape[1:])

        __inter_filters = x.shape[-1]

        phi_g = tf.keras.layers.Conv3D(__inter_filters, kernel_size = 1, strides = 1, padding = "same",
                                       kernel_initializer = "he_uniform", name = "phi_{}".format(__filters))(gated)
        theta_x = tf.keras.layers.Conv3D(__inter_filters, kernel_size = 1, strides = 1, padding = "same",
                                         kernel_initializer = "he_uniform", name = "theta_{}".format(__filters))(x)

        # theta_x_shape = theta_x.shape[1:-1]
        # phi_g_shape = phi_g.shape[1:-1]
        #
        # phi_g_upsampled = tf.keras.layers.UpSampling3D(tf.divide(theta_x_shape, phi_g_shape))(phi_g)
        phi_g_upsampled = tf.keras.layers.UpSampling3D([
            theta_x.shape[1] // phi_g.shape[1],
            theta_x.shape[2] // phi_g.shape[2],
            theta_x.shape[3] // phi_g.shape[3]
        ])(phi_g)

        __sum = tf.keras.layers.Add(name = "attention_add_{}".format(__filters))([phi_g_upsampled, theta_x])

        __activation_sum = tf.keras.layers.Activation("relu")(__sum)

        psi = tf.keras.layers.Conv3D(filters = 1, kernel_size = 1, strides = 1, padding = "same",
                                     kernel_initializer = "he_uniform", name = "psi_{}".format(__filters))(
            __activation_sum)

        __activation_psi = tf.keras.layers.Activation("sigmoid")(psi)

        __mul = tf.keras.layers.Multiply(name = "attention_multiplication_{}".format(__filters))([x, __activation_psi])

        # print("phi_g {}, phi_g_upsampled {}, theta_x {}, sum {}".format(phi_g.shape, phi_g_upsampled.shape, theta_x.shape, __sum.shape))
        # print("psi {}, mul {}".format(psi.shape, __mul.shape))

        __result = tf.keras.layers.Conv3D(filters = x.shape[-1], kernel_size = 1, strides = 1, padding = "same",
                                          kernel_initializer = "he_uniform")(__mul)

        if apply_batchnorm:
            __result = tf.keras.layers.BatchNormalization()(__result)

        # return tf.keras.models.Model(inputs = [input_x, input_g], outputs = [__result])
        return __result


if __name__ == "__main__":
    rnd = tf.random.uniform([3, 64, 64, 64, 8])

    inp = tf.keras.layers.Input(shape = rnd.shape[1:])
    x_shape = tf.keras.layers.Conv3D(16, kernel_size = 1, strides = 1, padding = "same", name="x")(inp)
    gated_shape = tf.keras.layers.Conv3D(128, kernel_size = (8, 8, 8), strides = (4, 4, 4), padding = "same", name="gate")(inp)

    result = AttGate(apply_batchnorm = False)((x_shape, gated_shape))

    model = tf.keras.models.Model(inputs = [inp], outputs = (result, x_shape, gated_shape))
    model.summary(line_length = 128, expand_nested = True, show_trainable = True)

    outputs = model(rnd)

    print("x {}, g {}, result {}".format(outputs[1].shape, outputs[2].shape, outputs[0].shape))
