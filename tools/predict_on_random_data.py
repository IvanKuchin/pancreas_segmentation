import time
import tensorflow as tf
import tools.config as config


def predict_on_random_data(model):
    for i in range(1, 32):
        t0 = time.time()
        pred = model.predict(
            tf.random.normal([i, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1]))
        t1 = time.time()
        print("prediction shape/time:", pred.shape, "/", np.round(t1 - t0))
