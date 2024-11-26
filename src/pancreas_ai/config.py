#################################
# specification of the model
#################################

EPOCHS = 1000
TRAIN_PASSES_PER_VALIDATION = 1

NUMBER_OF_CONV_IN_LAYER = 2

KERNEL_SIZE = [3,3,3]

INITIAL_LEARNING_RATE = 1e-4
BATCH_NORM = True
BATCH_SIZE = 1
BATCH_NORM_MOMENTUM = 0.8
DROPOUT = False
INSTANCE_NORM = False       # not supported yet

GRADIENT_ACCUMULATION_STEPS = 4 # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#args

LOSS_FUNCTION = "dice" # "dice" or "scce"

MONITOR_METRIC = "val_custom_f1"
MONITOR_MODE = "max"

MODEL_CHECKPOINT = "checkpoints/weights.keras"

TASK_TYPE = "classification" # "segmentation" or "classification"

################### DataSet ###################

VALIDATION_PERCENTAGE = 0.15

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

# segmentation
# IMAGE_DIMENSION_X = 160    
# IMAGE_DIMENSION_Y = IMAGE_DIMENSION_X
# IMAGE_DIMENSION_Z = IMAGE_DIMENSION_X

# classification
IMAGE_DIMENSION_X = 96
IMAGE_DIMENSION_Y = IMAGE_DIMENSION_X
IMAGE_DIMENSION_Z = IMAGE_DIMENSION_X

LABEL_SEGMENTATION_DIMENSION_X = IMAGE_DIMENSION_X
LABEL_SEGMENTATION_DIMENSION_Y = IMAGE_DIMENSION_Y
LABEL_SEGMENTATION_DIMENSION_Z = IMAGE_DIMENSION_Z

LABEL_CLASSIFICATION_DIMENSION = 1

AUGMENTATION_SHIFT_MARGIN = 0.1

MIN_LABEL = 0
MAX_LABEL = 1
MIN_DATA = -1
MAX_DATA = 1

CUTOUT_BORDER_FROM_PANCREAS = [90] # [0, 30, 60, 90]

IS_TILE = False

################### xxxxxxxxxxx ###################

# Dataset used for training
# consists of pickle files of 3d numpy arrays
TFRECORD_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/dataset/"

# DataSet recevied from POMC
# consists of dicom and nrrd files
POMC_PATIENTS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/local_training"
POMC_LABELS_SRC_FOLDER = "c:/Users/ikuchin/Downloads/pancreas_data/local_training"

######################################
##### Slice thicknes calculation #####
# https://stackoverflow.com/questions/21759013/dicom-affine-matrix-transformation-from-image-space-to-patient-space-in-matlab
THICKNESS = "strange"

# https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
# THICKNESS = "nibabel"

#################### Classification ##################
CLASSIFICATION_SEGMENTATION_MASK_FILENAME = "segmentation.nii.gz"
PANCREAS_ID_IN_MASK = 7

CLASSIFICATION_LABEL_FILENAME = "label"
#######################################
