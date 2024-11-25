import numpy as np
import os

class SaverNoTiled:
    def __init__(self, folder: str, subfolder: str, patient_id:str, percentage: int, config: dict):
        self.folder = folder
        self.subfolder = subfolder
        self.patient_id = patient_id
        self.percentage = percentage
        self.config = config
        self.image_dimension_x = config.IMAGE_DIMENSION_X
        self.image_dimension_y = config.IMAGE_DIMENSION_Y
        self.image_dimension_z = config.IMAGE_DIMENSION_Z

    def save(self, data, label):
        result = True
        
        with open(os.path.join(self.folder, self.subfolder, self.patient_id + f"_cut-{self.percentage}.npz", ), "wb") as f:
            np.savez_compressed(f, a=data, b=label)
            # np.savez_compressed(f, label)
        
        return result