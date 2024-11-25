import tensorflow as tf
import numpy as np
from dataset.savers.tiled import Slicer

DEBUG = False

class PredictTile:
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

    def scale_down(self, data):
        return data

    def scale_up(self, data):
        return data

    # input shape is [1, x, y, z, 1]
    # output shape is [x, y, z]
    def predict(self, src_data):
        src_data = tf.squeeze(src_data)
        fake_label = src_data
        result = np.zeros(src_data.shape)

        print("result shape is", result.shape)

        slicer = Slicer(src_data, fake_label, augment_margin=[0, 0, 0], config = self.config)
        for (data, _, x, y, z) in slicer:
            # data may be bigger than src_data shape
            # due to data padding in Slicer to the nearest multiple of IMAGE_DIMENSION_X, IMAGE_DIMENSION_Y, IMAGE_DIMENSION_Z
            # so we need to cut the data to the src_data shape
            if DEBUG == True:
                print(f"Predicting on a slice at {x}, {y}, {z} / shape {data.shape} ...")

            pred = self.model.predict(data[tf.newaxis, ..., tf.newaxis])
            pred = tf.argmax(pred, axis=-1)
            pred = tf.squeeze(pred)

            # cut the prediction to the src_data shape
            x_to = np.min([x + data.shape[0], result.shape[0]])
            y_to = np.min([y + data.shape[1], result.shape[1]])
            z_to = np.min([z + data.shape[2], result.shape[2]])
            if DEBUG == True:
                print(f"store in result {x}:{x_to}, {y}:{y_to}, {z}:{z_to}")

            # store the prediction in the result
            result[x:x_to, y:y_to, z:z_to] = pred[:x_to - x, :y_to - y, :z_to - z]

        return tf.constant(result)