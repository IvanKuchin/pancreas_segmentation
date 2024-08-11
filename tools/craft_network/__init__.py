import tools.craft_network.unet_classic
import tools.craft_network.unet_shortcuts_every_layer
import tools.craft_network.att_unet
import tools.craft_network.att_unet_dsv

import config as config


def craft_network(weights_file):
    return tools.craft_network.att_unet_dsv.craft_network(weights_file, apply_batchnorm = config.BATCH_NORM, apply_instancenorm = config.INSTANCE_NORM)
