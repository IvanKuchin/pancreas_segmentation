from . import classification
from . import segmentation

def ds_generator_factory(config: dict):
        if config.TASK_TYPE == "segmentation":
            return segmentation.Utils()
        elif config.TASK_TYPE == "classification":
            return classification.Utils()
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))
