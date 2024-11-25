import numpy as np
import unittest
from PIL import Image

import context
context.setup()
from pancreas_ai.dataset.ds_augmentation import rotate

class Rotate_3d(unittest.TestCase):
    '''
    Unit test to resize_3d function
    '''
    def test_rotate_90_deg(self):
        arr1 = np.arange(0, 3 * 3 * 2, dtype = np.int32)
        arr2 = np.reshape(arr1, [3, 3, -1])

        min = np.min(arr2)
        max = np.max(arr2)
        arr2 = (arr2 - min) / (max - min) * 2 - 1

        arr3 = rotate.rotate_data(90, 2, arr2)
        arr4 = rotate.rotate_data(-270, 2, arr2)

        self.assertEqual(np.all(arr3 == arr4), True, msg = "rotation 90 and -270 failed")

    def test_rotate_45_deg_images_axis_2(self):
        img1 = np.array(Image.open("tests/data/images/one.jpg"))
        img2 = np.array(Image.open("tests/data/images/two.jpg"))
        img3 = np.array(Image.open("tests/data/images/three.jpg"))

        # make it grayscale
        img1 = np.average(img1, axis = 2).astype(np.uint8)
        img2 = np.average(img2, axis = 2).astype(np.uint8)
        img3 = np.average(img3, axis = 2).astype(np.uint8)

        img4 = np.stack([img1, img2, img3], axis = 2)

        # normalize to [-1, 1]
        min = np.min(img4)
        max = np.max(img4)
        img4 = (img4 - min) / (max - min) * 2 - 1

        img5 = rotate.rotate_data(45, 2, img4)
        img6 = rotate.rotate_data(-(360-45), 2, img4)

        for i in range(3):
            # rescale back to [0, 255]
            img_to_save1 = img5[:, :, i] * 127.5 + 127.5
            img_to_save2 = img6[:, :, i] * 127.5 + 127.5

            rotated_img1 = Image.fromarray(img_to_save1.astype(np.uint8))
            rotated_img2 = Image.fromarray(img_to_save2.astype(np.uint8))

            rotated_img1.save(f"tests/data/images/rotated_{i}_45.jpg")
            rotated_img2.save(f"tests/data/images/rotated_{i}_{360-45}.jpg")

        self.assertAlmostEqual(np.sum(img5 - img6), 0, places=2, msg = "rotation 45 and -45 failed")


    def test_rotate_45_deg_images_axis_1(self):
        img1 = np.array(Image.open("tests/data/images/one.jpg"))
        img2 = np.array(Image.open("tests/data/images/two.jpg"))
        img3 = np.array(Image.open("tests/data/images/three.jpg"))

        # make it grayscale
        img1 = np.average(img1, axis = 2).astype(np.uint8)
        img2 = np.average(img2, axis = 2).astype(np.uint8)
        img3 = np.average(img3, axis = 2).astype(np.uint8)

        img4 = np.stack([img1, img2, img3], axis = 2)

        # normalize to [-1, 1]
        min = np.min(img4)
        max = np.max(img4)
        img4 = (img4 - min) / (max - min) * 2 - 1

        img5 = rotate.rotate_data(45, 0, img4)
        img6 = rotate.rotate_data(-(360-45), 0, img4)

        # saving images does not make sense here, due to rotation around axis 1

        self.assertAlmostEqual(np.sum(img5 - img6), 0, places=2, msg = "rotation 45 and -45 failed")




    # def test_resize_triple_and_back(self):
    #     arr1 = tf.range(0, 27, dtype = tf.int32)
    #     arr2 = tf.reshape(arr1, [3, 3, 3])

    #     arr3 = resize_3d.resize_3d_image(arr2, tf.constant([9, 9, 9]))
    #     arr4 = resize_3d.resize_3d_image(arr3, tf.constant([3, 3, 3]))

    #     result = tf.logical_not(tf.cast(tf.reshape(arr2-arr4, [-1]), dtype=tf.bool)).numpy().all()

    #     self.assertEqual(result, True, msg = "resize triple and back failed")

    # def test_resize_triple_and_back_xl(self):
    #     cube_size = 64
    #     arr2 = tf.random.uniform([cube_size, cube_size, cube_size], dtype = tf.float32)

    #     arr3 = resize_3d.resize_3d_image(arr2, tf.constant([cube_size*3, cube_size*3, cube_size*3]))
    #     arr4 = resize_3d.resize_3d_image(arr3, tf.constant([cube_size, cube_size, cube_size]))

    #     result = tf.logical_not(tf.cast(tf.reshape(arr2-arr4, [-1]), dtype=tf.bool)).numpy().all()

    #     self.assertEqual(result, True, msg = "resize triple and back failed")


if __name__ == "__main__":
    unittest.main()

