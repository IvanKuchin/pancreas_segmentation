EPOCHS = 1000

NUMBER_OF_CONV_IN_LAYER = 2

KERNEL_SIZE = [3,3,1]

BACKGROUND_WEIGHT = 1
FOREGROUND_WEIGHT = 7

INITIAL_LEARNING_RATE = 0.001
LOSS_SCALER = 100
BATCH_NORM = False
BATCH_SIZE = 1
BATCH_NORM_MOMENTUM = 0.8

PANCREAS_MIN_HU = -100024
PANCREAS_MAX_HU =  100024

IMAGE_DIMENSION_X = 160
IMAGE_DIMENSION_Y = IMAGE_DIMENSION_X
IMAGE_DIMENSION_Z = IMAGE_DIMENSION_X

MONITOR_METRIC = "val_custom_f1"
MONITOR_MODE = "max"

MODEL_CHECKPOINT = "checkpoints/weights.keras"

TFRECORD_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/original_npys/"

IMAGE_ORIGINAL_DIMENSION_X = 512
IMAGE_ORIGINAL_DIMENSION_Y = IMAGE_ORIGINAL_DIMENSION_X
IMAGE_ORIGINAL_DIMENSION_Z = IMAGE_ORIGINAL_DIMENSION_X