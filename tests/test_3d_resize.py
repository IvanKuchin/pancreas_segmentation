import tensorflow as tf
import unittest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d

class Resize_3d(unittest.TestCase):
    '''Unit test to resize_3d function'''
    def test_resize_to_same_size(self):
        arr1 = tf.range(0, 27, dtype = tf.int32)
        arr2 = tf.reshape(arr1, [3, 3, 3])

        arr3 = resize_3d.resize_3d_image(arr2, tf.constant([3, 3, 3]))
        result = tf.logical_not(tf.cast(tf.reshape(arr2-arr3, [-1]), dtype=tf.bool)).numpy().all()

        self.assertEqual(result, True, msg = "resize to the same size failed")

    def test_resize_twice_and_back(self):
        arr1 = tf.range(0, 27, dtype = tf.int32)
        arr2 = tf.reshape(arr1, [3, 3, 3])

        arr3 = resize_3d.resize_3d_image(arr2, tf.constant([6, 6, 6]))
        arr4 = resize_3d.resize_3d_image(arr3, tf.constant([3, 3, 3]))

        result = tf.logical_not(tf.cast(tf.reshape(arr2-arr4, [-1]), dtype=tf.bool)).numpy().all()

        self.assertEqual(result, True, msg = "resize twice and back failed")

    def test_resize_triple_and_back(self):
        arr1 = tf.range(0, 27, dtype = tf.int32)
        arr2 = tf.reshape(arr1, [3, 3, 3])

        arr3 = resize_3d.resize_3d_image(arr2, tf.constant([9, 9, 9]))
        arr4 = resize_3d.resize_3d_image(arr3, tf.constant([3, 3, 3]))

        result = tf.logical_not(tf.cast(tf.reshape(arr2-arr4, [-1]), dtype=tf.bool)).numpy().all()

        self.assertEqual(result, True, msg = "resize triple and back failed")

    def test_resize_triple_and_back_xl(self):
        cube_size = 64
        arr2 = tf.random.uniform([cube_size, cube_size, cube_size], dtype = tf.float32)

        arr3 = resize_3d.resize_3d_image(arr2, tf.constant([cube_size*3, cube_size*3, cube_size*3]))
        arr4 = resize_3d.resize_3d_image(arr3, tf.constant([cube_size, cube_size, cube_size]))

        result = tf.logical_not(tf.cast(tf.reshape(arr2-arr4, [-1]), dtype=tf.bool)).numpy().all()

        self.assertEqual(result, True, msg = "resize triple and back failed")


if __name__ == "__main__":
    unittest.main()

