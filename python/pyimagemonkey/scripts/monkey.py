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
from pyimagemonkey import MaskRcnnTrainer

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def _split_labels(labels, delimiter):
	return labels.split(delimiter)

def _create_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
		log.debug("Directory %s already exists...will create it" %(directory,))
	else:
		log.debug("Directory %s already exists...will use that one" %(directory,))



if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG) 

	parser = argparse.ArgumentParser(prog='PROG')
	subparsers = parser.add_subparsers(help='', dest='command')

	#add subparser for 'train'
	train_parser = subparsers.add_parser('train', help='train your own model')
	train_parser.add_argument('--labels', help='list of ImageMonkey labels that you want to train your model on', required=True)
	train_parser.add_argument("--delimiter", help="label delimiter", default="|")
	train_parser.add_argument("--type", help="type (object_detection | image_classification | object_segmentation)", required=True)
	
	#add subparser for 'list-labels'
	list_labels_parser = subparsers.add_parser('list-labels', help='list all labels that are available at ImageMonkey')

	args = parser.parse_args()

	imagemonkey_api = API(api_version=1)


	if args.command == "train":
		train_type = None
		directory = ""
		if args.type == "object_detection":
			train_type = Type.OBJECT_DETECTION
			directory = "/tmp/object_detection/"
		elif args.type == "image_classification":
			train_type = Type.IMAGE_CLASSIFICATION
			directory = "/tmp/image_classification/"
		elif args.type == "object_segmentation":
			train_type = args.type
			directory = "/tmp/object_segmentation/"
		else:
			print("Unknown object_detection type: %s" %(train_type,))
			sys.exit(1)

		_create_dir_if_not_exists(directory)

		if train_type == Type.OBJECT_DETECTION or train_type == Type.IMAGE_CLASSIFICATION:
			try:
				tensorflow_trainer = TensorflowTrainer(directory, clear_before_start=True, tf_object_detection_models_path="/tensorflow_models/")
				tensorflow_trainer.train(_split_labels(args.labels, args.delimiter), min_probability = 0.8, train_type = train_type)
			except Exception as e: 
				print(e)
		else:
			maskrcnn_trainer = MaskRcnnTrainer(directory, model="/home/imagemonkey/models/resnet/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
			maskrcnn_trainer.train(_split_labels(args.labels, args.delimiter), min_probability = 0.8)


	if args.command == "list-labels":
		labels = imagemonkey_api.labels()
		for label in labels:
			print(label)