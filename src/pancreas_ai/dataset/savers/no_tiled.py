import numpy as np
import math
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import resize_3d
import src.pancreas_ai.config as config

class SaverNoTiled:
    def __init__(self, folder: str, subfolder: str, patient_id:str, percentage: int, image_dimension_x: int, image_dimension_y: int, image_dimension_z: int):
        self.folder = folder
        self.subfolder = subfolder
        self.patient_id = patient_id
        self.percentage = percentage
        self.image_dimension_x = image_dimension_x
        self.image_dimension_y = image_dimension_y
        self.image_dimension_z = image_dimension_z

    def save(self, data, label):
        result = True
        
        with open(os.path.join(self.folder, self.subfolder, self.patient_id + f"_cut-{self.percentage}.npz", ), "wb") as f:
            np.savez_compressed(f, a=data, b=label)
            # np.savez_compressed(f, label)
        
        return result