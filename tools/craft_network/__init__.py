import tools.craft_network.unet_classic
import tools.craft_network.unet_shortcuts_every_layer


def craft_network(weights_file):
    print("craft_network: " + weights_file)
    return tools.craft_network.unet_shortcuts_every_layer.craft_network(weights_file)
