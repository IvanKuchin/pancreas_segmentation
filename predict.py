import tensorflow as tf
import numpy as np
import time
import os

def print_model_weights(model):
    model.summary()
    for layer in model.layers:
        print(layer.name)
        print(layer.weights)


def main():
    model = tf.keras.models.load_model("pancreas_segmentation_model.h5")
    # print_model_weights(model)


if __name__ == "__main__":
    main()

