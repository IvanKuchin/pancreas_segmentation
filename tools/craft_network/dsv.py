import tensorflow as tf


class DSV(tf.keras.layers.Layer):
    def __init__(self, scale_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.classes = 2 # 1 - foreground, 1 - background
        # print("DSV mode constructor (input shape in unknown)")


    def build(self, inputs):
        # print("building model {}".format(inputs))

        self.conv = tf.keras.layers.Conv3D(self.classes, kernel_size = 1, padding = "same",
                                            kernel_initializer = "he_uniform", name="dsv_conv")
        self.upsample = tf.keras.layers.UpSampling3D([self.scale_factor, self.scale_factor, 1], name="dsv_upsample")

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)

        if self.scale_factor > 1:
            x = self.upsample(x)

        return x

    ##########################
    # ATTENTION !!! 
    # having this function will cause the model to fail to load weights from file
    ##########################
    # def compute_output_shape(self, input_shape):
    #     (b, h, w, d, _) = input_shape
    #     return tf.TensorShape((b, h * self.scale_factor, w * self.scale_factor, d, self.classes))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "scale_factor": self.scale_factor}

if __name__ == "__main__":
    rnd = tf.random.uniform([3, 64, 64, 256, 16])

    inp = tf.keras.layers.Input(shape = rnd.shape[1:])
    x_shape = tf.keras.layers.Conv3D(16, kernel_size = 1, strides = 1, padding = "same")(inp)
    dsv_model = DSV(scale_factor = 4)
    result = dsv_model(x_shape)

    model = tf.keras.models.Model(inputs = [inp], outputs = [result])
    outputs = model(rnd)

    model.summary()

    print("model.outputs {}".format(model.outputs[0]))
    print("result {}".format(outputs.shape))
