from ds_generator import classification
from ds_generator import segmentation

def ds_generator_factory(type:str):
        if type == "segmentation":
            return segmentation.Utils()
        elif type == "classification":
            return classification.Utils()
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))