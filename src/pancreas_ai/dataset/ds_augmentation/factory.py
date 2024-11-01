from . import segmentation
from . import classification

def augment_factory(type:str):
    if type == "segmentation":
        return segmentation.Augment()
    elif type == "classification":
        return classification.Augment()
    else:
        raise ValueError("Unknown crop factory type: {}".format(type))