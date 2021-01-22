LOSS_SCALER = 100
BATCH_NORM = False
BATCH_SIZE = 1
BATCH_NORM_MOMENTUM = 0.8

PANCREAS_MIN_HU = -1024
PANCREAS_MAX_HU =  1024

IMAGE_DIMENSION_X = 64
IMAGE_DIMENSION_Y = IMAGE_DIMENSION_X
IMAGE_DIMENSION_Z = IMAGE_DIMENSION_X

MODEL_CHECKPOINT = "weights.hdf5"

TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/{}x{}x{}/".format(IMAGE_DIMENSION_X, IMAGE_DIMENSION_Y, IMAGE_DIMENSION_Z)
