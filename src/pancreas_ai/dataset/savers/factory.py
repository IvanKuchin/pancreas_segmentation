from .tiled import SaverTiled
from .no_tiled import SaverNoTiled

def saver_factory(config: dict):
    if config.IS_TILE:
        return SaverTiled
    elif config.IS_TILE == False:
        return SaverNoTiled
    else:
        raise ValueError("Unknown saver type: {}".format(type))
