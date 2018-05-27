#!/usr/bin/python3

import argparse
import logging
import sys
import os
import io
sys.path.insert(1, os.path.join(sys.path[0], ('..' + os.path.sep + "..")))

from pyimagemonkey import API
from pyimagemonkey import TensorflowTrainer
from pyimagemonkey import Type

def _split_labels(labels, delimiter):
	return labels.split(delimiter)



if __name__ == "__main__":
	logging.basicConfig()

	parser = argparse.ArgumentParser()
	parser.add_argument('--labels', help='labels that you want to train your model on', required=True)
	parser.add_argument("--delimiter", help="label delimiter", default="|")
	parser.add_argument("--detection_type", help="type (object_detection | image_classification)", default="image_classification")
	args = parser.parse_args()

	detection_type = None
	if args.detection_type == "object_detection":
		detection_type = Type.OBJECT_DETECTION
	elif args.detection_type == "image_classification":
		detection_type = Type.IMAGE_CLASSIFICATION
	else:
		print("Unknown object_detection type: %s" %(detection_type,))
		sys.exit(1)


	#["orange", "spoon"]
	tensorflow_trainer = TensorflowTrainer("/tmp/test", clear_before_start=True, tf_object_detection_models_path="/tensorflow_models/")
	tensorflow_trainer.train(_split_labels(args.labels, args.delimiter), min_probability = 0.8, train_type = detection_type)