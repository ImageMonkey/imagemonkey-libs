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
	logging.basicConfig(level=logging.DEBUG)

	parser = argparse.ArgumentParser(prog='PROG')
	subparsers = parser.add_subparsers(help='', dest='command')

	#add subparser for 'train'
	train_parser = subparsers.add_parser('train', help='train your own model')
	train_parser.add_argument('--labels', help='list of ImageMonkey labels that you want to train your model on', required=True)
	train_parser.add_argument("--delimiter", help="label delimiter", default="|")
	train_parser.add_argument("--detection_type", help="type (object_detection | image_classification)", default="image_classification")
	
	#add subparser for 'list-labels'
	list_labels_parser = subparsers.add_parser('list-labels', help='list all labels that are available at ImageMonkey')

	args = parser.parse_args()

	imagemonkey_api = API(api_version=1)


	if args.command == "train":
		detection_type = None
		if args.detection_type == "object_detection":
			detection_type = Type.OBJECT_DETECTION
		elif args.detection_type == "image_classification":
			detection_type = Type.IMAGE_CLASSIFICATION
		else:
			print("Unknown object_detection type: %s" %(detection_type,))
			sys.exit(1)


		try:
			tensorflow_trainer = TensorflowTrainer("/tmp/test", clear_before_start=True, tf_object_detection_models_path="/tensorflow_models/")
			tensorflow_trainer.train(_split_labels(args.labels, args.delimiter), min_probability = 0.8, train_type = detection_type)
		except Exception as e: 
			print(e)

	if args.command == "list-labels":
		labels = imagemonkey_api.labels()
		for label in labels:
			print(label)