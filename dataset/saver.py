import numpy as np
import math
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d
import config as config

class Slicer:
    def __init__(self, data, label, augment_margin = [0, 0, 0]):
        x = math.ceil(data.shape[0] / config.IMAGE_DIMENSION_X) * config.IMAGE_DIMENSION_X
        y = math.ceil(data.shape[1] / config.IMAGE_DIMENSION_Y) * config.IMAGE_DIMENSION_Y
        z = math.ceil(data.shape[2] / config.IMAGE_DIMENSION_Z) * config.IMAGE_DIMENSION_Z

        self.data = np.zeros((x, y, z)) + config.MIN_DATA
        self.label = np.zeros((x, y, z)) + config.MIN_LABEL

        self.data[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        self.label[:label.shape[0], :label.shape[1], :label.shape[2]] = label

        self.augment_margin = augment_margin

    def __iter__(self):
        for x in range(0, self.data.shape[0], config.IMAGE_DIMENSION_X):
            for y in range(0, self.data.shape[1], config.IMAGE_DIMENSION_Y):
                for z in range(0, self.data.shape[2], config.IMAGE_DIMENSION_Z):
                    x_start = np.max([x - self.augment_margin[0], 0])
                    y_start = np.max([y - self.augment_margin[1], 0])
                    z_start = np.max([z - self.augment_margin[2], 0])

                    x_finish = np.min([x + config.IMAGE_DIMENSION_X + self.augment_margin[0], self.data.shape[0]])
                    y_finish = np.min([y + config.IMAGE_DIMENSION_Y + self.augment_margin[1], self.data.shape[1]])
                    z_finish = np.min([z + config.IMAGE_DIMENSION_Z + self.augment_margin[2], self.data.shape[2]])

                    data  = self.data [x_start:x_finish, y_start:y_finish, z_start:z_finish]
                    label = self.label[x_start:x_finish, y_start:y_finish, z_start:z_finish]

                    if np.max(label) == 0:
                        continue

                    yield data, label, x, y, z


class Saver:
    def __init__(self, folder: str, subfolder: str, patient_id:str, percentage: int, image_dimension_x: int, image_dimension_y: int, image_dimension_z: int):
        self.folder = folder
        self.subfolder = subfolder
        self.patient_id = patient_id
        self.percentage = percentage
        self.image_dimension_x = image_dimension_x
        self.image_dimension_y = image_dimension_y
        self.image_dimension_z = image_dimension_z

    def save(self, src_data, label_data):
        result = True
        src_data = np.cast[np.float32](src_data)
        label_data = np.cast[np.int8](label_data)

        augment_margin = [
            int(config.IMAGE_DIMENSION_X * config.AUGMENTATIO_SHIFT_MARGIN), 
            int(config.IMAGE_DIMENSION_Y * config.AUGMENTATIO_SHIFT_MARGIN), 
            int(config.IMAGE_DIMENSION_Z * config.AUGMENTATIO_SHIFT_MARGIN)
            ]


        for (data, label, x, y, z) in Slicer(src_data, label_data, augment_margin=augment_margin):
            # print(f"Saving slice at {x}, {y}, {z}...")
            np.savez_compressed(os.path.join(self.folder, self.subfolder, self.patient_id + f"_cut-{self.percentage}_slice-{x}-{y}-{z}.npz", ), [data, label])

        return result
