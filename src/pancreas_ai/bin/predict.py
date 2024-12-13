import sys
import os

import numpy as np

from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai.tools import resize_3d
from pancreas_ai.dataset.pomc_reader.factory import reader_factory
from pancreas_ai import config

class Predict:
    def __init__(self, config: dict):
        self.config = config
        self.ct_folder = os.path.join(os.getcwd(), self.config.PREDICTION_CT_FOLDER)
        self.mask_file = os.path.join(self.ct_folder, self.config.CLASSIFICATION_SEGMENTATION_MASK_FILENAME)

        self.reader = reader_factory(config.TASK_TYPE, self.config)

    def is_ct_folder_exists(self):        
        if not(os.path.exists(self.ct_folder)):
            return False
        
        # check if the folder contains CT files
        for root, dirs, files in os.walk(self.ct_folder):
            for file in files:
                if file.endswith(".dcm"):
                    return True

        return True
    
    def segment(self):
        segmentation = totalsegmentator(self.ct_folder, fastest=False)
        nib.save(segmentation, self.mask_file)

    def cancer_probability(self):
        data, data_metadata = self.reader.read_data(self.ct_folder)

        # substitute fake label
        if not self.reader.check_before_processing(data, np.array([0]), data_metadata, {}):
            return

        scaled_data = resize_3d.resize_3d_image(data, np.array([self.config.IMAGE_DIMENSION_X, self.config.IMAGE_DIMENSION_Y, self.config.IMAGE_DIMENSION_Z]))

        result = scaled_data

        return result


def main():
    predict = Predict(config)

    print("Searching for a folder with CT-scan", end="")
    if predict.is_ct_folder_exists():
        print("\tok") 
    else:
        print("\tfail")
        return

    print("Segmentation")
    # predict.segment()

    print("Mask pancreas")
    pred = predict.cancer_probability()


if __name__ == "__main__":
    main()