import os
import sys
import numpy as np
from shapes import ShapesConfig, ShapesDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = os.path.abspath('../../')
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import visualize
import mrcnn.model as modellib



config = ShapesConfig()
config.display()

dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Load and display random samples
"""
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
"""


# Create model in training mode
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# Train the head branches 
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

#model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='all')

# dection
