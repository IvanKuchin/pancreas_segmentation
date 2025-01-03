import os
import numpy as np
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai import config
from pancreas_ai.tools.craft_network import factory


def layer_info(layer):
    theta = layer.theta
    weights = theta.get_weights()
    print("Layer name: ", layer.name)
    # print("Layer input shape: ", layer.input_shape)
    # print("Layer output shape: ", layer.output_shape)
    print("Layer config: ", layer.get_config())

def main():
    checkpoint_file = "test.keras"
    weights_file = "test.weights.h5"
    config.MODEL_CHECKPOINT = ""

    model_original = factory.model_factory(config)

    inp = tf.ones(shape=(3, config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z, 1))
    y_original = model_original.predict(inp)

    model_original.summary(line_length=128)
    print("Model output shape: ", y_original.shape)
 
    model_original.save(checkpoint_file)
    model_original.save_weights(weights_file)

    config.MODEL_CHECKPOINT = checkpoint_file
    model_reconstructed = factory.model_factory(config)
    y_reconstructed = model_reconstructed.predict(inp)
    print(f"diff after crafting the model from {checkpoint_file}: ", np.sum(y_reconstructed - y_original))
    model_reconstructed.load_weights(weights_file)
    y_reconstructed = model_reconstructed.predict(inp)
    print(f"diff after load_weights from {weights_file}: ", np.sum(y_reconstructed - y_original))

    os.remove(weights_file)
    os.remove(checkpoint_file)

if __name__ == "__main__":
    main()
