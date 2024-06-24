import tensorflow as tf
import numpy as np

def test_custom_loss():
    y_true = np.array([[[[[  1], [  0]], [[  0], [  0]]], [[[  0], [  0]], [[  0], [  0]]]]])
    y_pred = np.array([[[[[0.1, 0.9], [0.9, 0.1]], [[0.1, 0.9], [0.9, 0.1]]], [[[0.1, 0.9], [0.9, 0.1]], [[0.1, 0.9], [0.9, 0.1]]]]])
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    loss = __custom_loss(y_true, y_pred)
    assert loss.numpy() == 0.105360515

test_custom_loss()