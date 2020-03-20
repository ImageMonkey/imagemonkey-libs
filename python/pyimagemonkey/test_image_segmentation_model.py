#import skimage
#from pyimagemonkey.mask_rcnn_config import ImageMonkeyConfig
import pyimagemonkey.helper as helper
#from mrcnn import model as modellib
#from mrcnn import visualize

# import keras
import keras

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

class TestMaskRcnnModel(object):
    def __init__(self, path_to_model, path_to_labels, output_dir, clear_before_start=False):
        self._path_to_model = path_to_model
        self._path_to_labels = path_to_labels
        self._output_dir = output_dir

        if clear_before_start:
            helper.clear_output_dir(self._output_dir)

    def _load_labels(self):
        labels = {}

        i = 0
        with open(self._path_to_labels, "r") as f:
            lines = f.readlines() 
            for line in lines:
                labels[i] = line
                i += 1
        return labels
                

    def annotate_image(self, path_to_image, num_gpus=1, min_image_dimension=800, max_image_dimension=1024, 
                steps_per_epoch = 100, validation_steps = 70):
        
        labels_to_names = self._load_labels()

        model = models.load_model(self._path_to_model, backbone_name='resnet50')

        # load image
        image = read_image_bgr(path_to_image)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks  = outputs[-1][0]

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < 0.5:
                break

            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            mask = mask[:, :, label]
            draw_mask(draw, b, mask, color=label_color(label))
            
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()


        """config = ImageMonkeyConfig(len(labels), num_gpus, min_image_dimension, max_image_dimension, steps_per_epoch, validation_steps) 
        config.display()

        model = modellib.MaskRCNN(mode="inference", model_dir="/tmp", config=config)
        model.load_weights(self._path_to_model, by_name=True) 
        image = skimage.io.imread(path_to_image)

        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            labels, r['scores'])
        """
