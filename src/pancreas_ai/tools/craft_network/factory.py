from . import att_unet_dsv
from . import classification

def model_factory(config):
    """
    Factory function to create a model based on the configuration
    """
    if config.TASK_TYPE == "segmentation":
        return att_unet_dsv.craft_network(config)
    elif config.TASK_TYPE == "classification":
        return classification.craft_network(config)
    else:
        raise ValueError("Model type {} not supported".format(config.MODEL_TYPE))