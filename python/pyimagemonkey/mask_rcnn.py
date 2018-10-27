import os
import sys
import json
import datetime
import numpy as np
import skimage
import shutil
import math
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util

from pyimagemonkey.api import *
from pyimagemonkey.exceptions import *

from mrcnn.config import Config
from mrcnn import model as modellib, utils


class ImageMonkeyConfig(Config):

    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "imagemonkey"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    #IMAGE_MIN_DIM = 256 #800
    #IMAGE_MAX_DIM = 320 #1024

    def __init__(self, num_classes, num_gpus, min_image_dimension, max_image_dimension, 
                    steps_per_epoch, validation_steps):
        #set NUM_CLASSES before calling base class
        #otherwise it won't work
        self.NUM_CLASSES = num_classes + 1 #(num of classes +1 for background)
        self.GPU_COUNT = num_gpus
        self.IMAGE_MIN_DIM = min_image_dimension
        self.IMAGE_MAX_DIM = max_image_dimension
        self.STEPS_PER_EPOCH = steps_per_epoch #Number of training steps per epoch
        self.VALIDATION_STEPS = validation_steps #Number of validation steps
        super().__init__() 

class ImageMonkeyDataset(utils.Dataset):

    def _add_me(self, ctr, mode):
        m = 3 #split up dataset; 30% validation set, 70% training set
        if (ctr%3) == 0:
            if mode == "validation":
                return True
        else:
            if mode == "training":
                return True
        return False

    def load(self, entries, labels, mode):
        """Load a subset of the ImageMonkey dataset.
        subset: Subset to load: train or val
        """
        # Add classes.

        i = 1
        for label in labels:
            self.add_class("imagemonkey", i, label)
            i += 1
        self._all_labels = labels

        # Train or validation dataset?
        assert mode in ["training", "validation"]
        

        ctr = 0
        for entry in entries:
            ctr += 1
            if self._add_me(ctr, mode):
                img = entry.image

                annotations = entry.annotations
                self.add_image(
                    "imagemonkey",
                    image_id=img.uuid,
                    path=img.path,
                    width=img.width,
                    height=img.height,
                    annotations=annotations,
                    labels=labels
                )

    def load_mask(self, image_id):
        # If not a ImageMonkey dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "imagemonkey":
            return super(self.__class__, self).load_mask(image_id)

        log.debug("Loading mask for image with id %s" %(image_id))

        class_ids = []
        annotations = image_info["annotations"]
        #create n-dimensional mask with zeros
        mask = np.zeros([image_info["width"], image_info["height"], len(annotations)],
                        dtype=np.uint8)
        for i, annotation in enumerate(annotations):
            if annotation.label not in self._all_labels:
                raise ImageMonkeyGeneralError("Warning: imagemonkey annotation with label %s as not in labels to train" %(annotation.label))

            if type(annotation.data) is Ellipse:
                trimmed_ellipse = annotation.data.trim(Rectangle(0, 0, image_info["width"], image_info["height"]))


                rr, cc = skimage.draw.ellipse(trimmed_ellipse.left + trimmed_ellipse.rx, trimmed_ellipse.top + trimmed_ellipse.ry, 
                                              trimmed_ellipse.rx, trimmed_ellipse.ry, rotation=math.radians(trimmed_ellipse.angle))
                mask[rr, cc, i] = 1


            elif type(annotation.data) is Rectangle or type(annotation.data) is Polygon:
                polypoints = annotation.data.points
                trimmed_polypoints = polypoints.trim(Rectangle(0, 0, image_info["width"], image_info["height"])) 

                xvals = []
                yvals = []
                for polypoint in trimmed_polypoints.points:
                    xvals.append(polypoint.x)
                    yvals.append(polypoint.y)

                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(xvals, yvals)
                mask[rr, cc, i] = 1

            class_ids.append(self.class_names.index(annotation.label)) #get class id from label

        #if there is at least one annotation
        if class_ids:
            class_ids = np.array(class_ids, dtype=np.int32)

            return mask, class_ids

        # Call super class to return an empty mask (in case there are no annotations)
        return super(self.__class__, self).load_mask(image_id) 




    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "imagemonkey":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class Trainer(object):
    def __init__(self, training_dir, clear_before_start):
        self._training_dir = training_dir
        self._images_dir = training_dir + os.path.sep + "images"
        self._models_dir = training_dir + os.path.sep + "models"
        self._checkpoints_dir = training_dir + os.path.sep + "checkpoints"
        self._api = API(api_version=1)

        if clear_before_start:
            if os.path.exists(self._images_dir):
                self.clear_images_dir()

        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        if not os.path.exists(self._models_dir):
            os.makedirs(self._models_dir)

        if not os.path.exists(self._checkpoints_dir):
            os.makedirs(self._checkpoints_dir)

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    def clear_images_dir(self):
        for f in os.listdir(self._images_dir):
            filePath = os.path.join(self._images_dir, f)
            try:
                if os.path.isfile(filePath):
                    os.unlink(filePath)
                elif os.path.isdir(filePath): shutil.rmtree(filePath)
            except Exception as e:
                log.error("Couldn't clear images directory %s" %(self._images_dir,))
                raise ImageMonkeyGeneralError("Couldn't clear images directory %s" %(self._images_dir,)) 

    def _create_category_dir(self, category):
        path = self._images_dir + os.path.sep + category
        if os.path.exists(path):
            log.info("Directory %s already exists!" %(path,))
            raise ImageMonkeyGeneralError("Directory %s already exists!" %(path,)) 
        
        os.makedirs(path)

    def _category_dir_exists(self, category):
        path = self._images_dir + os.path.sep + category
        if os.path.exists(path):
            return True
        return False


class MaskRcnnTrainer(Trainer):
    def __init__(self, training_dir, clear_before_start=True, model="imagenet"):
        super(MaskRcnnTrainer, self).__init__(training_dir, clear_before_start)
        self._training_dataset = ImageMonkeyDataset() #training dataset
        self._validation_dataset = ImageMonkeyDataset() #validation dataset

        self._base_model = model
        

 
    def _export_data_and_download_images(self, labels, min_probability):
        extension = ".jpg"

        #for the MaskRcnnTrainer implementation we can store the images all in one folder
        dir_name = "default"
        if not self._category_dir_exists(dir_name):
            self._create_category_dir(dir_name)

        folder = self._images_dir + os.path.sep + dir_name

        res = self._api.export(labels, min_probability, only_annotated=True)
        for elem in res:
            path = folder + os.path.sep + elem.image.uuid + extension
            elem.image.path = path
            elem.image.folder = folder
            self._api.download_image(elem.image.uuid, folder, extension=extension)

        return res

    def save_model_to_pb(self):
        log.info("Saving model...")
        # Create model in inference mode
        saved_model = modellib.MaskRCNN(mode="inference",
                                    config=self._config, model_dir=self.checkpoints_dir)
        model_path = saved_model.find_last()
        log.debug("Loading weights from %s" %(model_path,))
        saved_model.load_weights(model_path, by_name=True)

        # All new operations will be in test mode from now on.
        K.set_learning_phase(0)

        filename = os.path.splitext(os.path.basename(model_path))[0] + ".pb"
        sess = K.get_session() # get the tensorflow session
        model_keras = saved_model.keras_model
        output_names_all = [output.name.split(':')[0] for output in model_keras.outputs]

        # Getthe graph to export
        graph_to_export = sess.graph

        # Freeze the variables in the graph and remove heads that were not selected
        # this will also cause the pb file to contain all the constant weights
        od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                graph_to_export.as_graph_def(),
                                                                output_names_all)

        model_dirpath = os.path.dirname(model_path)
        pb_filepath = os.path.join(model_dirpath, filename)
        log.debug('Saving frozen graph %s...' %(os.path.basename(pb_filepath)))
        frozen_graph_path = pb_filepath
        with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
            f.write(od_graph_def.SerializeToString())
        log.info("Froze graph: %s" %(pb_filepath))

    def train(self, labels, min_probability=0.8, num_gpus=1, 
                min_image_dimension=800, max_image_dimension=1024, 
                steps_per_epoch = 100, validation_steps = 70, 
                epochs = 30, save_best_only = True):
        self._config = ImageMonkeyConfig(len(labels), num_gpus, min_image_dimension, max_image_dimension, 
                                    steps_per_epoch, validation_steps)
        self._config.display()

        self._model = modellib.MaskRCNN(mode="training", config=self._config,
                                        model_dir=self.checkpoints_dir)


        model_path = None
        if self._base_model == "imagenet":
            model_path = self._model.get_imagenet_weights()
        else:
            model_path = self._base_model

        if model_path is None:
            raise ImageMonkeyGeneralError("Model path is missing - please provide a valid model path!")

        data = self._export_data_and_download_images(labels, min_probability)

        log.debug("Loading weights %s" %(model_path,))
        self._model.load_weights(model_path, by_name=True)
        
        self._training_dataset.load(data, labels, "training")
        self._training_dataset.prepare()

        self._validation_dataset.load(data, labels, "validation")
        self._validation_dataset.prepare()


        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        log.info("Training network heads")
        self._model.train(self._training_dataset, self._training_dataset,
                    learning_rate=self._config.LEARNING_RATE,
                    epochs=epochs, layers='heads',
                    save_best_only=save_best_only)

        self.save_model_to_pb()

