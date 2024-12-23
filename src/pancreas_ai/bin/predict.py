import sys
import os

import numpy as np
import tensorflow as tf

from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai import config
from pancreas_ai.dataset.pomc_reader.factory import reader_factory
from pancreas_ai.tools import resize_3d
from pancreas_ai.tools.craft_network import factory
from pancreas_ai.dataset.pomc_reader.preprocess import preprocess_data

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
        segmentation = totalsegmentator(self.ct_folder, fastest=False, verbose=True)
        nib.save(segmentation, self.mask_file)

    def cancer_probability(self) -> np.array:
        data, data_metadata = self.reader.read_data(self.ct_folder)
        fake_label = np.array([0])

        if not self.reader.check_before_processing(data, fake_label, data_metadata, {}):
            return

        data_shape = np.array([self.config.IMAGE_DIMENSION_X, self.config.IMAGE_DIMENSION_Y, self.config.IMAGE_DIMENSION_Z])
        data = resize_3d.resize_3d_image(data, data_shape)
        data, fake_label = preprocess_data(data, fake_label, config)
        if self.reader.check_after_preprocessing(data, fake_label) == False:
            print("ERROR: data or label failed sanity check")
            return

        model = factory.model_factory(config)

        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=-1)
        tensor = tf.convert_to_tensor(data, dtype=tf.float32)

        pred = model.predict(tensor)

        return pred


def main():
    predict = Predict(config)

    print("Searching for a folder with CT-scan", end="")
    if predict.is_ct_folder_exists():
        print("\tok") 
    else:
        print("\tfail")
        return

    print("Segmentation")
    predict.segment()

    print("Pancreas cancer probability calculation...")
    pred = predict.cancer_probability()
    pred = pred[0][0]
    print("probaility of cancer:", np.round(pred, 2))
    if pred > 0.7:
        print("Cancer detected")
    if pred < 0.3:
        print("Cancer not detected")



if __name__ == "__main__":
    main()
