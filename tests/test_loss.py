import unittest
import numpy as np
import tensorflow as tf

from tools.craft_network.loss import loss_func_generator

class Loss(unittest.TestCase):
    def __exp1_pred1_1(self, loss: str):
        y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
        y_pred = np.array([[[[[0.1, 4.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        # print("shape of y_true: ", y_true.shape)
        # print("shape of y_pred: ", y_pred.shape)

        loss = loss_func_generator(loss)(y_true, y_pred)
        # print("expect 1, predict 1: loss {:.5f}".format(loss.numpy()))

        return loss

    def __exp1_pred1_2(self, loss: str):
        y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
        y_pred = np.array([[[[[0.1, 0.9], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        loss = loss_func_generator(loss)(y_true, y_pred)
        # print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

        return loss

    def __exp1_pred0(self, loss: str):
        y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
        y_pred = np.array([[[[[4.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        loss = loss_func_generator(loss)(y_true, y_pred)
        # print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

        return loss

    def __exp0_pred1(self, loss: str):
        y_true = np.array([[[[     [  1],      [  0]], [     [  0],      [  0]]], [[     [  0],      [  0]], [     [  0],      [  0]]]]])
        y_pred = np.array([[[[[0.1, 0.9], [0.1, 4.9]], [[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.1], [0.9, 0.1]], [[0.9, 0.1], [0.9, 0.1]]]]])
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        loss = loss_func_generator(loss)(y_true, y_pred)
        # print("expect 1, predict 1: loss {:.4f}".format(loss.numpy()))

        return loss

    def __big_volume_exp1_pred1(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_true[0, 10:11, 10:11, 10:11, 0] = 1
        y_pred[0, 10:11, 10:11, 10:11, 1] = 1
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        loss = loss_func_generator(loss)(y_true, y_pred)

        return loss

    def __big_volume_exp1_pred05(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_true[0, 10:11, 10:11, 10:11, 0] = 1.0
        y_pred[0, 10:11, 10:11, 10:11, 1] = 0.5
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        loss = loss_func_generator(loss)(y_true, y_pred)
    
        return loss

    def __big_volume_exp1_pred0(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_true[0, 10:11, 10:11, 10:11, 0] = 1.0
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        loss = loss_func_generator(loss)(y_true, y_pred)
    
        return loss

    def __big_volume_exp2x1_pred0(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_true[0, 10:12, 10:11, 10:11, 0] = 1.0
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        loss = loss_func_generator(loss)(y_true, y_pred)
    
        return loss

    def __big_volume_exp0_pred1(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_pred[0, 10:11, 10:11, 10:11, 1] = 1
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        loss = loss_func_generator(loss)(y_true, y_pred)
    
        return loss

    def __big_volume_exp0_pred2x1(self, loss: str):
        mx_size = 200
        y_true = np.zeros((1, mx_size, mx_size, mx_size, 1))
        y_pred = np.zeros((1, mx_size, mx_size, mx_size, 2))
        y_pred[0, 10:12, 10:11, 10:11, 1] = 1
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        loss = loss_func_generator(loss)(y_true, y_pred)
    
        return loss


    def test_exp1_pred1_dice(self):
        loss1 = self.__exp1_pred1_1("dice")
        loss2 = self.__exp1_pred1_2("dice")
        self.assertGreater(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_exp1_pref1_scce(self):
        loss1 = self.__exp1_pred1_1("scce")
        loss2 = self.__exp1_pred1_2("scce")
        self.assertLess(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_exp1_pred0_dice(self):
        loss1 = self.__exp1_pred0("dice")
        loss2 = self.__exp1_pred1_1("dice")
        self.assertGreater(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_exp1_pred0_scce(self):
        loss1 = self.__exp0_pred1("scce")
        loss2 = self.__exp1_pred1_2("scce")
        self.assertGreater(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_big_volume_exp1_pred1_dice(self):
        loss1 = self.__big_volume_exp1_pred1("dice")
        self.assertAlmostEqual(loss1.numpy(), 0, places=5)

    def test_big_volume_exp1_pred1_scce(self):
        loss1 = self.__big_volume_exp1_pred1("scce")
        self.assertAlmostEqual(loss1.numpy(), 0, places=5)

    def test_big_volume_exp1_pred_less_scce(self):
        loss1 = self.__big_volume_exp1_pred05("scce")
        loss2 = self.__big_volume_exp1_pred0("scce")
        self.assertLess(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_big_volume_exp1_pred_less_scce(self):
        loss1 = self.__big_volume_exp1_pred05("dice")
        loss2 = self.__big_volume_exp1_pred0("dice")
        self.assertLess(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_big_volume_exp2x1_exp1_scce(self):
        loss1 = self.__big_volume_exp2x1_pred0("scce")
        loss2 = self.__big_volume_exp1_pred0("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), np.abs(loss2.numpy()), places=5)

    def test_big_volume_pred2x1_pred1_scce(self):
        loss1 = self.__big_volume_exp0_pred1("scce")
        loss2 = self.__big_volume_exp0_pred2x1("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), np.abs(loss2.numpy()), places=5)

    def test_big_volume_exp2x1_exp1_dice(self):
        loss1 = self.__big_volume_exp2x1_pred0("dice")
        loss2 = self.__big_volume_exp1_pred0("dice")
        self.assertGreater(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_big_volume_pred2x1_pred1_dice(self):
        loss1 = self.__big_volume_exp2x1_pred0("dice")
        loss2 = self.__big_volume_exp1_pred0("dice")
        self.assertGreater(np.abs(loss1.numpy()), np.abs(loss2.numpy()))

    def test_big_volume_scce_almost0_1(self):
        loss1 = self.__big_volume_exp0_pred1("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), 0, places=5)

    def test_big_volume_scce_almost0_2(self):
        loss1 = self.__big_volume_exp0_pred2x1("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), 0, places=5)

    def test_big_volume_scce_almost0_3(self):
        loss1 = self.__big_volume_exp1_pred0("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), 0, places=5)

    def test_big_volume_scce_almost0_4(self):
        loss1 = self.__big_volume_exp2x1_pred0("scce")
        self.assertAlmostEqual(np.abs(loss1.numpy()), 0, places=5)

    def test_big_volume_dice_almost0_1(self):
        loss1 = self.__big_volume_exp0_pred1("dice")
        self.assertGreater(np.abs(loss1.numpy()), 0)

    def test_big_volume_dice_almost0_2(self):
        loss1 = self.__big_volume_exp0_pred2x1("dice")
        self.assertGreater(np.abs(loss1.numpy()), 0)

    def test_big_volume_dice_almost0_3(self):
        loss1 = self.__big_volume_exp1_pred0("dice")
        self.assertGreater(np.abs(loss1.numpy()), 0)

    def test_big_volume_dice_almost0_4(self):
        loss1 = self.__big_volume_exp2x1_pred0("dice")
        self.assertGreater(np.abs(loss1.numpy()), 0)

if __name__ == "__main__":
    unittest.main()

