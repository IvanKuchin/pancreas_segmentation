import unittest
import numpy as np

import context
context.setup()
from pancreas_ai.tools import resize_3d

class Resize_3d(unittest.TestCase):
    '''
    Unit test to resize_3d function
    '''
    def test_resize_to_same_size(self):
        arr1 = np.arange(0, 27)
        arr2 = np.reshape(arr1, [3, 3, -1])

        min = np.min(arr2)
        max = np.max(arr2)
        arr2 = (arr2 - min) / (max - min) * 2 - 1

        arr3 = resize_3d.resize_3d_image(arr2, np.array([3, 3, 3]))
        result = np.all(arr2 == arr3)

        self.assertEqual(result, True, msg = "resize to the same size failed")

    def test_resize_twice_and_back(self):
        arr1 = np.arange(0, 27, dtype = np.int32)
        arr2 = np.reshape(arr1, [3, 3, -1])

        min = np.min(arr2)
        max = np.max(arr2)
        arr2 = (arr2 - min) / (max - min) * 2 - 1

        arr3 = resize_3d.resize_3d_image(arr2, np.array([6, 6, 6]))
        arr4 = resize_3d.resize_3d_image(arr3, np.array([3, 3, 3]))

        result = np.all(arr2 == arr4)

        self.assertAlmostEqual(result, 0, places = 5, msg = "resize twice and back failed")

    def test_resize_triple_and_back(self):
        arr1 = np.arange(0, 27, dtype = np.int32)
        arr2 = np.reshape(arr1, [3, 3, -1])

        arr3 = resize_3d.resize_3d_image(arr2, np.array([9, 9, 9]))
        arr4 = resize_3d.resize_3d_image(arr3, np.array([3, 3, 3]))

        result = np.all(arr2 == arr4)

        self.assertAlmostEqual(result, 0, places = 5, msg = "resize triple and back failed")

    def test_resize_triple_and_back_xl(self):
        cube_size = 64
        arr2 = np.random.randn(cube_size, cube_size, cube_size)

        arr3 = resize_3d.resize_3d_image(arr2, np.array([cube_size*3, cube_size*3, cube_size*3]))
        arr4 = resize_3d.resize_3d_image(arr3, np.array([cube_size, cube_size, cube_size]))

        result = np.all(arr2 == arr4)

        self.assertAlmostEqual(result, 0, places = 5, msg = "resize triple and back failed")


if __name__ == "__main__":
    unittest.main()

