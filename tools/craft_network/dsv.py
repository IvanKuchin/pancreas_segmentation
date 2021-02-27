import tensorflow as tf


class DSV(tf.keras.layers.Layer):
    def __init__(self, scale_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.classes = 2 # 1 - foreground, 1 - background


    def build(self, inputs):
        # print("{} {}".format())

        self.conv = tf.keras.layers.Conv3D(self.classes, kernel_size = 1, padding = "same",
                                            kernel_initializer = "he_uniform")

        self.upsample = tf.keras.layers.UpSampling3D([self.scale_factor, self.scale_factor, 1])

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        # print("theta_x {}".format(theta_x.shape))
        x = self.upsample(x)
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[:-1] + [2])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "scale_factor": self.scale_factor}

if __name__ == "__main__":
    rnd = tf.random.uniform([3, 64, 64, 256, 16])

    inp = tf.keras.layers.Input(shape = rnd.shape[1:])
    x_shape = tf.keras.layers.Conv3D(16, kernel_size = 1, strides = 1, padding = "same")(inp)
    # gated_shape = tf.keras.layers.Conv3D(128, kernel_size = (8, 8, 1), strides = (1, 1, 1), padding = "same")(inp)
    # gated_shape = tf.keras.layers.Conv3D(128, kernel_size = (8, 8, 1), strides = (4, 4, 1), padding = "same")(inp)

    # result = attention_gate(x = x, gated = gated, apply_batchnorm = False)
    result = DSV(scale_factor = 2)(x_shape)

    model = tf.keras.models.Model(inputs = [inp], outputs = [result])
    outputs = model(rnd)

    print("result {}".format(outputs.shape))
