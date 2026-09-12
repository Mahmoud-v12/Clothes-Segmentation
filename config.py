"""
Central configuration for the clothes segmentation pipeline.

"""

import tensorflow as tf

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
DATASET_NAME = "mattmdjaga/human_parsing_dataset"

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

NUM_CLASSES = 18

VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 1e-3

AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------------------------------------------
# Class-weight estimation (used to upweight rare classes in the loss)
# ------------------------------------------------------------------
CLASS_WEIGHT_MAX_SAMPLES = 1000

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 7

# ------------------------------------------------------------------
# Output paths
# ------------------------------------------------------------------
BEST_MODEL_PATH = "best_unet.keras"
FINAL_MODEL_PATH = "human_parsing_unet_18_classes.keras"
