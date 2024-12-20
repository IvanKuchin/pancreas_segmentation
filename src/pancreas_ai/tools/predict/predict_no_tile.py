import tensorflow as tf
from tools import resize_3d
import src.pancreas_ai.config as config

class PredictNoTile:
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

    def scale_down(self, data):
        self.shape = data.shape
        data = resize_3d.resize_3d_image(data,
                                         tf.constant([
                                            self.config.IMAGE_DIMENSION_X, 
                                            self.config.IMAGE_DIMENSION_Y, 
                                            self.config.IMAGE_DIMENSION_Z
                                        ]))
        data = tf.cast(data, tf.float32)

        return data

    def scale_up(self, data):
        data = resize_3d.resize_3d_image(data, tf.constant(self.shape))
        return data

    # input shape is [1, x, y, z, 1]
    # output shape is [x, y, z]
    def predict(self, src_data):
        pred = self.model.predict(src_data)
        pred = tf.argmax(pred, axis = -1)
        pred = tf.squeeze(pred)
        return pred
