import tensorflow as tf
import unittest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import tools.craft_network.att_unet_dsv as att_unet_dsv
import tools.craft_network.att_unet as att_unet

class Load_Save_Models(unittest.TestCase):
    def __save_weights_load_weights(self, model_constructor, file_name):
        print("build network1")
        orig_model = model_constructor()

        input_shape = orig_model.input_shape
        input = tf.ones(shape = (3,) + input_shape[1:])
        pred_original = orig_model.predict(input)

        print("save weigths to", file_name)
        orig_model.save_weights(file_name)

        print("build network2")
        reconst_model = model_constructor()
        _ = reconst_model(input)
        print("load weigths from", file_name)
        reconst_model.load_weights(file_name)
        pred_reconst = reconst_model.predict(input)

        os.remove(file_name)

        return pred_original, pred_reconst

    def __save_model_load_weights(self, model_constructor, file_name):
        print("build network1")
        orig_model = model_constructor()

        input_shape = orig_model.input_shape
        input = tf.ones(shape = (3,) + input_shape[1:])
        pred_original = orig_model.predict(input)

        print("save weigths to", file_name)
        orig_model.save(file_name)

        print("build network2")
        reconst_model = model_constructor()
        _ = reconst_model(input)
        print("load weigths from", file_name)
        reconst_model.load_weights(file_name)
        pred_reconst = reconst_model.predict(input)

        os.remove(file_name)

        return pred_original, pred_reconst

    def test_att_unet_dsv_save_weights_load_weights(self):
        model_builder = att_unet_dsv.craft_network
        fname = "unit_test_att_unet_dsv.weights.h5"
        pred_original, pred_reconstructed = self.__save_weights_load_weights(model_builder, fname)

        self.assertAlmostEqual(tf.reduce_sum(tf.abs(pred_original - pred_reconstructed)).numpy(), 0, places = 5)

    def test_att_unet_save_weights_load_weights(self):
        model_builder = att_unet.craft_network
        fname = "unit_test_att_unet_dsv.weights.h5"
        pred_original, pred_reconstructed = self.__save_weights_load_weights(model_builder, fname)

        self.assertAlmostEqual(tf.reduce_sum(tf.abs(pred_original - pred_reconstructed)).numpy(), 0, places = 5)

    def test_att_unet_dsv_save_model_load_weights(self):
        model_builder = att_unet_dsv.craft_network
        fname = "unit_test_att_unet_dsv.keras"
        pred_original, pred_reconstructed = self.__save_model_load_weights(model_builder, fname)

        self.assertAlmostEqual(tf.reduce_sum(tf.abs(pred_original - pred_reconstructed)).numpy(), 0, places = 5)

    def test_att_unet_save_model_load_weights(self):
        model_builder = att_unet.craft_network
        fname = "unit_test_att_unet_dsv.keras"
        pred_original, pred_reconstructed = self.__save_model_load_weights(model_builder, fname)

        self.assertAlmostEqual(tf.reduce_sum(tf.abs(pred_original - pred_reconstructed)).numpy(), 0, places = 5)

if __name__ == "__main__":
    unittest.main()
