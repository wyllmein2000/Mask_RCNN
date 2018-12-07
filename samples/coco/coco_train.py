import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

COCO_DIR = '/home/momo/wu.yan/coco'
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


# MS COCO Dataset
import coco
config = coco.CocoConfig()


# Load dataset
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, 'train')
dataset_train.prepare()

dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, 'val')
dataset_val.prepare()

print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))




DEVICE = '/device:GPU:0'
MODE = 'training'

# Create model in training mode
# with tf.device(DEVICE):
model = modellib.MaskRCNN(mode=MODE, config=config, model_dir=MODEL_DIR)

#model_path = COCO_MODEL_PATH
model_path = model.find_last()

model.load_weights(model_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # Train the head branches 
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
