import numpy as np
import math
import os

class Slicer:
    def __init__(self, data, label, augment_margin = [0, 0, 0], config: dict = {}):
        self.config = config
        
        x = math.ceil(data.shape[0] / self.config.IMAGE_DIMENSION_X) * self.config.IMAGE_DIMENSION_X
        y = math.ceil(data.shape[1] / self.config.IMAGE_DIMENSION_Y) * self.config.IMAGE_DIMENSION_Y
        z = math.ceil(data.shape[2] / self.config.IMAGE_DIMENSION_Z) * self.config.IMAGE_DIMENSION_Z

        self.data = np.zeros((x, y, z)) + self.config.MIN_DATA
        self.label = np.zeros((x, y, z)) + self.config.MIN_LABEL

        self.data[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        self.label[:label.shape[0], :label.shape[1], :label.shape[2]] = label

        self.augment_margin = augment_margin

    def __iter__(self):
        for x in range(0, self.data.shape[0], self.config.IMAGE_DIMENSION_X):
            for y in range(0, self.data.shape[1], self.config.IMAGE_DIMENSION_Y):
                for z in range(0, self.data.shape[2], self.config.IMAGE_DIMENSION_Z):
                    x_start = np.max([x - self.augment_margin[0], 0])
                    y_start = np.max([y - self.augment_margin[1], 0])
                    z_start = np.max([z - self.augment_margin[2], 0])

                    x_finish = np.min([x + self.config.IMAGE_DIMENSION_X + self.augment_margin[0], self.data.shape[0]])
                    y_finish = np.min([y + self.config.IMAGE_DIMENSION_Y + self.augment_margin[1], self.data.shape[1]])
                    z_finish = np.min([z + self.config.IMAGE_DIMENSION_Z + self.augment_margin[2], self.data.shape[2]])

                    data  = self.data [x_start:x_finish, y_start:y_finish, z_start:z_finish]
                    label = self.label[x_start:x_finish, y_start:y_finish, z_start:z_finish]

                    if np.max(label) == 0:
                        continue

                    yield data, label, x, y, z


class SaverTiled:
    def __init__(self, folder: str, subfolder: str, patient_id:str, percentage: int, config: dict):
        self.folder = folder
        self.subfolder = subfolder
        self.patient_id = patient_id
        self.percentage = percentage
        self.config = config
        self.image_dimension_x = config.IMAGE_DIMENSION_X
        self.image_dimension_y = config.IMAGE_DIMENSION_Y
        self.image_dimension_z = config.IMAGE_DIMENSION_Z

    def save(self, src_data, label_data):
        result = True
        src_data = np.cast[np.float32](src_data)
        label_data = np.cast[np.int8](label_data)

        augment_margin = [
            int(self.config.IMAGE_DIMENSION_X * self.config.AUGMENTATION_SHIFT_MARGIN), 
            int(self.config.IMAGE_DIMENSION_Y * self.config.AUGMENTATION_SHIFT_MARGIN), 
            int(self.config.IMAGE_DIMENSION_Z * self.config.AUGMENTATION_SHIFT_MARGIN)
            ]


        for (data, label, x, y, z) in Slicer(src_data, label_data, augment_margin=augment_margin, config = self.config):
            # print(f"Saving slice at {x}, {y}, {z}...")
            with open(os.path.join(self.folder, self.subfolder, self.patient_id + f"_cut-{self.percentage}_slice-{x}-{y}-{z}.npz", ), "wb") as f:
                np.savez_compressed(f, data)
                np.savez_compressed(f, label)

        return result
