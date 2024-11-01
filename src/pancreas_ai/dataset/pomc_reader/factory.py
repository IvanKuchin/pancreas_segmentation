from . import classification
from . import segmentation

def reader_factory(type:str, config: dict):
        if type == "segmentation":
            return segmentation.Reader(config)
        elif type == "classification":
            return classification.Reader(config)
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))