import skimage
from pyimagemonkey.mask_rcnn_config import ImageMonkeyConfig
import pyimagemonkey.helper as helper
from mrcnn import model as modellib
from mrcnn import visualize

class TestMaskRcnnModel(object):
    def __init__(self, path_to_model, path_to_labels, output_dir, clear_before_start=False):
        self._path_to_model = path_to_model
        self._path_to_labels = path_to_labels
        self._output_dir = output_dir

        if clear_before_start:
            helper.clear_output_dir(self._output_dir)

    def _load_labels(self):
        return ["cat", "dog"]

    def annotate_image(self, path_to_image, num_gpus=1, min_image_dimension=800, max_image_dimension=1024, 
                steps_per_epoch = 100, validation_steps = 70):
        labels = self._load_labels()
        config = ImageMonkeyConfig(len(labels), num_gpus, min_image_dimension, max_image_dimension, steps_per_epoch, validation_steps) 
        config.display()

        model = modellib.MaskRCNN(mode="inference", model_dir="/tmp", config=config)
        model.load_weights(self._path_to_model, by_name=True) 
        image = skimage.io.imread(path_to_image)

        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            labels, r['scores'])
