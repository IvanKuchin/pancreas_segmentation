from dataset.pomc_reader import classification
from dataset.pomc_reader import segmentation

def reader_factory(type:str):
        if type == "segmentation":
            return segmentation.Reader
        elif type == "classification":
            return classification.Reader
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))