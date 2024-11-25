from . import classification
from . import segmentation

def ds_generator_factory(config: dict):
        if config.TASK_TYPE == "segmentation":
            return segmentation.Utils([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z])
        elif config.TASK_TYPE == "classification":
            return classification.Utils([config.LABEL_CLASSIFICATION_DIMENSION])
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))
