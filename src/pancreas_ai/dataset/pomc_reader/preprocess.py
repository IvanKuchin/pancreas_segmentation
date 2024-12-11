import numpy as np
import numpy.typing as npt

def preprocess_data(data: npt.NDArray[np.float32], label: npt.NDArray[np.float32], config) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # zoom = AUGMENT_SCALED_DIMS / data.shape
    # data_zoomed = scipy.ndimage.interpolation.zoom(data, zoom, mode="nearest")
    # label_zoomed = scipy.ndimage.interpolation.zoom(label, zoom, mode="nearest")

    #
    # output minHU/maxHU of pancreas area
    #
    # gt_idx = label == 1
    # min_HU = np.min(data[gt_idx])
    # max_HU = np.max(data[gt_idx])
    # print("minTotal/minHU/maxHU/maxTotal: {}/{}/{}/{}".format(np.min(data), min_HU, max_HU, np.max(data)))


    #
    # Restrict CT voxel values to [pancreas HU], this will give wider range to pancreas,
    # compare to original data [pancreas HU]
    #
    data_idx1 = data <= config.PANCREAS_MIN_HU
    data_idx2 = data >= config.PANCREAS_MAX_HU

    data[data_idx1] = config.PANCREAS_MIN_HU
    data[data_idx2] = config.PANCREAS_MAX_HU

    #
    # Assign -1 to mask that is outside of pancreas HU
    #
    # label[data_idx1] = -1
    # label[data_idx2] = -1

    # data_zoomed = resize_3d.resize_3d_image(data, AUGMENT_SCALED_DIMS)
    # label_zoomed = resize_3d.resize_3d_image(label, AUGMENT_SCALED_DIMS)

    # self.print_statistic(label, label_zoomed)

    #
    # scale final data to [-1; 1] range, that should help with ReLU activation
    #
    spread = config.MAX_DATA - config.MIN_DATA
    data_processed = (data - config.PANCREAS_MIN_HU) / (config.PANCREAS_MAX_HU - config.PANCREAS_MIN_HU) * spread - spread / 2
    # if data_processed.shape != AUGMENT_SCALED_DIMS:
    #     print_error("wrong Z-axis dimensionality {} must be {}".format(data_processed.shape, AUGMENT_SCALED_DIMS))

    return data_processed, label
