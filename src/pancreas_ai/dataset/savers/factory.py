from dataset.savers.tiled import SaverTiled
from dataset.savers.no_tiled import SaverNoTiled

class SaverFactory:
    def __call__(self, type:str):
        if type == "tiled":
            return SaverTiled
        elif type == "no_tiled":
            return SaverNoTiled
        else:
            raise ValueError("Unknown saver type: {}".format(type))