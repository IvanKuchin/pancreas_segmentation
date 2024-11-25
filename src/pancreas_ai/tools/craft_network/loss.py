import tensorflow as tf

def __dice_coef(y_true, y_pred):
    gamma = 1.0
    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred[..., 1:2], dtype = tf.float32)

    # print("y_true shape: ", y_true.shape)
    # print("y_pred shape: ", y_pred.shape)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + gamma) / (union + gamma)
    return dice

def __dice_loss(y_true, y_pred):
    return 1 - __dice_coef(y_true, y_pred)


def __weighted_loss(y_true, y_pred):
    scaler = 1

    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred, dtype = tf.float32)

    # count_0 = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    # count_1 = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))
    # background_weight = (1 - count_0 / (count_0 + count_1)) * config.LOSS_SCALER
    # foreground_weight = (1 - count_1 / (count_0 + count_1)) * config.LOSS_SCALER / 5

    # background_weight = config.BACKGROUND_WEIGHT  # incorrect loss calculations
    # foreground_weight = config.FOREGROUND_WEIGHT  # incorrect loss calculations
    # foreground_weight -= background_weight  # incorrect loss calculations

    # loss function described in the paper https://arxiv.org/pdf/1803.05431v2
    foreground_size = tf.reduce_sum(tf.cast(y_true == 1.0, y_true.dtype))
    background_size = tf.reduce_sum(tf.cast(y_true == 0.0, y_true.dtype))
    size = tf.cast(tf.size(y_true), dtype = tf.float32)
    foreground_weight = (1.0 - foreground_size / size) * scaler
    background_weight = (1.0 - background_size / size)
    # foreground and background weights must add up to 1 
    # it will be added/reversed later
    foreground_weight -= background_weight

    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    loss = scce(
        tf.maximum(y_true, 0.0),  # remove -1 values from mask,
        y_pred,
        sample_weight = tf.maximum(y_true * foreground_weight + background_weight, 0.0),
    )

    return loss

def __segmentation_loss(config: dict):
    if config.LOSS_FUNCTION == "dice":
        return __dice_loss
    elif config.LOSS_FUNCTION == "scce":
        return __weighted_loss
    else:
        raise ValueError("Unknown loss function")


def __classification_loss(config: dict):
    return tf.keras.losses.BinaryCrossentropy(from_logits = False)


def loss_func_factory(config: dict):
    if config.TASK_TYPE == "segmentation":
        return __segmentation_loss(config)
    elif config.TASK_TYPE == "classification":
        return __classification_loss(config)
    else:
        raise ValueError("Unknown loss function")

