import os
import sys
import json
import datetime
import numpy as np
import skimage
import shutil
import math
import cv2 as cv
import tensorflow as tf
import pyimagemonkey.helper as helper

from keras import backend as K
from tensorflow.python.framework import graph_util

from pyimagemonkey.api import *
from pyimagemonkey.exceptions import *

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pyimagemonkey.mask_rcnn_config import ImageMonkeyConfig

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

                rr, cc = skimage.draw.ellipse(trimmed_ellipse.cx, trimmed_ellipse.cy, 
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
        self._statistics_dir = self._training_dir + os.path.sep + "statistics"
        self._output_dir = self._training_dir + os.path.sep + "output"

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

        if not os.path.exists(self._statistics_dir):
            os.makedirs(self._statistics_dir)

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    @property
    def output_dir(self):
        return self._output_dir

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
    def __init__(self, training_dir, clear_before_start=True, model="imagenet", filter_dataset=None, statistics=None):
        super(MaskRcnnTrainer, self).__init__(training_dir, clear_before_start)
        self._filter = filter_dataset
        self._statistics = statistics
        self._base_model = model
        self._all_labels = []
        
        self._tmp_output_dir = self.output_dir + os.path.sep + "tmp"
        self._classes_file = self._tmp_output_dir + os.path.sep + "classes.txt"
        self._annotations_file =  self._tmp_output_dir + os.path.sep + "annotations.csv"
        self._mask_output_dir = self.output_dir + os.path.sep + "tmp" + os.path.sep + "masks"

        if not os.path.exists(self._tmp_output_dir):
            os.makedirs(self._tmp_output_dir)
    
        if not os.path.exists(self._mask_output_dir):
            os.makedirs(self._mask_output_dir)    

 
    def _export_data_and_download_images(self, labels, min_probability):
        extension = ".jpg"

        #for the MaskRcnnTrainer implementation we can store the images all in one folder
        dir_name = "default"
        if not self._category_dir_exists(dir_name):
            self._create_category_dir(dir_name)

        folder = self._images_dir + os.path.sep + dir_name

        res = self._api.export(labels, min_probability, only_annotated=True)
        if self._filter is not None:
            res = self._filter.filter(res)

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

    

    def _create_classes_file(self, labels):
        out = ""
        ctr = 0
        for label in labels:
            out += label + "," + str(ctr) + "\n"
            ctr += 1
        with open(self._classes_file, "w") as f:
            f.write(out)

    def _create_annotations_file(self, entries):
        out = ""
        for entry in entries:
            annotations = entry.annotations
            image_width = entry.image.width
            image_height = entry.image.height

            # Create a black image
            mask_img = np.zeros((image_height, image_width, 3), np.uint8)
            
            for i, annotation in enumerate(annotations):
                if annotation.label not in self._all_labels:
                    raise ImageMonkeyGeneralError("Warning: imagemonkey annotation with label %s as not in labels to train" %(annotation.label))
              
                mask_output_path = self._mask_output_dir + os.path.sep + entry.image.uuid + "_" + str(i) + ".png" 
                bounding_box = None 

                if type(annotation.data) is Ellipse:
                    trimmed_ellipse = annotation.data.trim(Rectangle(0, 0, image_width, image_height))
                    #TODO

                elif type(annotation.data) is Rectangle or type(annotation.data) is Polygon:
                    polypoints = annotation.data.points
                    trimmed_polypoints = polypoints.trim(Rectangle(0, 0, image_width, image_height))

                    p = []
                    xvals = []
                    yvals = []
                    for polypoint in trimmed_polypoints.points:
                        p.append([polypoint.x, polypoint.y])
                        xvals.append(polypoint.x)
                        yvals.append(polypoint.y)

                    cv.fillPoly(mask_img, pts =[np.asarray(p)], color=(255,255,255))
                
                    bounding_box = [min(xvals), min(yvals), max(xvals), max(yvals)]
                
                cv.imwrite(mask_output_path, mask_img)
                print(annotation.label) 
                out += (entry.image.path + "," + str(bounding_box[0]) + "," + str(bounding_box[1]) 
                    + "," + str(bounding_box[2]) + "," + str(bounding_box[3]) + "," + annotation.label + "," + mask_output_path + "\n")
        
        with open(self._annotations_file, "w") as f:
            f.write(out)
    
    def train(self, labels, min_probability=0.8, num_gpus=1, 
                min_image_dimension=800, max_image_dimension=1024, 
                steps_per_epoch = 100, validation_steps = 70, 
                epochs = 30, save_best_only = True):

        self._all_labels = labels
        model_path = None
        if self._base_model == "imagenet":
            model_path = self._model.get_imagenet_weights()
        else:
            model_path = self._base_model

        if model_path is None:
            raise ImageMonkeyGeneralError("Model path is missing - please provide a valid model path!")

        data = self._export_data_and_download_images(labels, min_probability)

        self._create_classes_file(labels)
        self._create_annotations_file(data)

        cmd = "maskrcnn-train csv " + self._annotations_file + " " + self._classes_file
        print(cmd)
        
        helper.run_command(cmd) 

        """if self._statistics is not None:
            self._statistics.output_path = self._statistics_dir + os.path.sep + "statistics.json"
            #self._statistics.class_names = self._training_dataset.class_names
            self._statistics.generate(data)
            self._statistics.save()
        """

