from . import segmentation
from . import classification

def augment_factory(config: dict):
    if config.TASK_TYPE == "segmentation":
        return segmentation.Augment(config)
    elif config.TASK_TYPE == "classification":
        return classification.Augment(config)
    else:
        raise ValueError("Unknown crop factory type: {}".format(config.TASK_TYPE))
    