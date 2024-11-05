import time
import tensorflow as tf
import numpy as np


def predict_on_random_data(model, config):
    for i in range(2, 3):
        t0 = time.time()
        pred = model.predict(
            tf.random.normal([i, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1]))
        t1 = time.time()
        print("prediction shape/time:", pred.shape, "/", np.round(t1 - t0))
