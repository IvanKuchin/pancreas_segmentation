EPOCHS = 1000
TRAIN_PASSES_PER_VALIDATION = 1

NUMBER_OF_CONV_IN_LAYER = 2

KERNEL_SIZE = [3,3,3]

# BACKGROUND_WEIGHT = 1  # must be calculated dynamically
# FOREGROUND_WEIGHT = 7  # must be calculated dynamically

INITIAL_LEARNING_RATE = 1e-4
INSTANCE_NORM = False       # not supported yet
BATCH_NORM = True
BATCH_SIZE = 1
BATCH_NORM_MOMENTUM = 0.8

GRADIENT_ACCUMULATION_STEPS = 4 # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#args

# Option 1) HU range for pancreas in CT scans from 30 to 400
# https://radiopaedia.org/articles/windowing-ct?lang=us
# Option 2) 3D Slicer preset for abdominal CT
# W/L: 350/40, which makes the pancreas range from -310 to 390
#
# Our pancreas calculations show values from -1200 to 4000
#
# training attempts shows that the best performance in the range [-512, 1024]
PANCREAS_MIN_HU =  -512    # -512
PANCREAS_MAX_HU =  1024    # 1024

IMAGE_DIMENSION_X = 160
IMAGE_DIMENSION_Y = IMAGE_DIMENSION_X
IMAGE_DIMENSION_Z = IMAGE_DIMENSION_X

AUGMENTATION_SHIFT_MARGIN = 0.1

MIN_LABEL = 0
MAX_LABEL = 1
MIN_DATA = -1
MAX_DATA = 1

MONITOR_METRIC = "val_custom_f1"
MONITOR_MODE = "max"

MODEL_CHECKPOINT = "checkpoints/weights.keras"

IMAGE_ORIGINAL_DIMENSION_X = 512
IMAGE_ORIGINAL_DIMENSION_Y = IMAGE_ORIGINAL_DIMENSION_X
IMAGE_ORIGINAL_DIMENSION_Z = IMAGE_ORIGINAL_DIMENSION_X

IS_TILE = False

# Dataset used for training
# consists of pickle files of 3d numpy arrays
TFRECORD_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/dataset/"

# DataSet recevied from POMC
# consists of dicom and nrrd files
POMC_PATIENTS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/local_training"
POMC_LABELS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/local_training"
