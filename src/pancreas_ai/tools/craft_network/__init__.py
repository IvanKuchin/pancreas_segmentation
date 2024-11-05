# from . import unet_classic
# from . import unet_shortcuts_every_layer
from . import att_unet
from . import att_unet_dsv

def craft_network(config: dict):
    return att_unet_dsv.craft_network(config)
