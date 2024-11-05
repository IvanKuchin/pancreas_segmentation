import numpy as np

def __crop(data, offsrt, size):
    return data[
            offsrt[0]:offsrt[0] + size[0],
            offsrt[1]:offsrt[1] + size[1],
            offsrt[2]:offsrt[2] + size[2],
            ...]

def random_crop_data_and_label(data, label, x, y, z) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop data and label 
    """
    data_shape = np.shape(data)
    random_range = [data_shape[0] - x + 1, data_shape[1] - y + 1, data_shape[2] - z + 1]
    random_offset = np.random.randint(0, random_range, size = 3)
    
    _data = __crop(data, random_offset, [x, y, z])
    _label = __crop(label, random_offset, [x, y, z])
    return _data, _label

def random_crop_data(data, inside_width, inside_height, inside_depth) -> np.ndarray:
    data_shape = np.shape(data)
    random_range = [
        (data_shape[0] - inside_width + 1) / 2, 
        (data_shape[1] - inside_height + 1) / 2, 
        (data_shape[2] - inside_depth + 1) / 2]
    random_offset = np.random.randint(0, random_range, size = 3)

    return __crop(data, random_offset, [inside_width, inside_height, inside_depth])

