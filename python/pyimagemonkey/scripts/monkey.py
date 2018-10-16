#!/usr/bin/python3

import argparse
import logging
import sys
import os
import io
sys.path.insert(1, os.path.join(sys.path[0], ('..' + os.path.sep + "..")))

from pyimagemonkey import API
from pyimagemonkey import Type
from pyimagemonkey import TensorflowTrainer
from pyimagemonkey import MaskRcnnTrainer

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def _split_labels(labels, delimiter):
	return labels.split(delimiter)

def _create_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
		log.debug("Directory %s doesn't exist...will create it" %(directory,))
	else:
		log.debug("Directory %s already exists...will use that one" %(directory,))



if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG) 

	parser = argparse.ArgumentParser(prog='PROG')
	subparsers = parser.add_subparsers(help='', dest='command')

	#add subparser for 'train'
	train_parser = subparsers.add_parser('train', help='train your own model')
	train_parser.add_argument('--labels', help='list of ImageMonkey labels that you want to train your model on. default: all labels', default=None, required=False)
	train_parser.add_argument("--delimiter", help="label delimiter", default="|")
	train_parser.add_argument("--type", help="type (object-detection | image-classification | object-segmentation)", required=True)
	train_parser.add_argument("--num-gpus", help="number of GPUs you want to use for training", required=False, default=None, type=int)
	train_parser.add_argument("--min-img-size", help="minimal image size", required=False, default=None, type=int)
	train_parser.add_argument("--max-img-size", help="maximal image size", required=False, default=None, type=int)
	train_parser.add_argument("--steps-per-epoch", help="training steps per epoch", required=False, default=None, type=int)
	train_parser.add_argument("--validation-steps", help="validation steps", required=False, default=None, type=int)
	train_parser.add_argument("--learning-rate", help="learning rate", required=False, default=None)

	#add subparser for 'list-labels'
	list_labels_parser = subparsers.add_parser('list-labels', help='list all labels that are available at ImageMonkey')

	args = parser.parse_args()

	imagemonkey_api = API(api_version=1)


	if args.command == "train":
		train_type = None
		directory = ""
		if args.type == "object-detection":
			train_type = Type.OBJECT_DETECTION
			directory = "/tmp/object_detection/"
		elif args.type == "image-classification":
			train_type = Type.IMAGE_CLASSIFICATION
			directory = "/tmp/image_classification/"
		elif args.type == "object-segmentation":
			train_type = args.type
			directory = "/tmp/object_segmentation/"
		else:
			print("Unknown object_detection type: %s" %(train_type,))
			sys.exit(1)

		_create_dir_if_not_exists(directory)

		labels = None
		if args.labels is None:
			labels = imagemonkey_api.labels(True)
		else:
			labels = _split_labels(args.labels, args.delimiter)

		if labels is None:
			print("Please provide a labels list first!")
			sys.exit(1)


		if train_type == Type.OBJECT_DETECTION or train_type == Type.IMAGE_CLASSIFICATION:
			if args.num_gpus is not None:
				parser.error('--num-gpus is only allowed when --type=object-segmentation')
			if args.min_img_size is not None:
				parser.error('--min-img-size is only allowed when --type=object-segmentation')
			if args.max_img_size is not None:
				parser.error('--max-img-size is only allowed when --type=object-segmentation')
			if args.steps_per_epoch is not None:
				parser.error('--steps-per-epoch is only allowed when --type=object-segmentation')
			if args.validation_steps is not None:
				parser.error('--validation-steps is only allowed when --type=object-segmentation')
			if (args.learning_rate is not None) and (train_type == Type.IMAGE_CLASSIFICATION):
				parser.error('--learning-rate is only allowed when --type=object-detection')

			try:
				tensorflow_trainer = TensorflowTrainer(directory, clear_before_start=True, tf_object_detection_models_path="/root/tensorflow_models/")
				tensorflow_trainer.train(labels, min_probability = 0.8, train_type = train_type, learning_rate=args.learning_rate)
			except Exception as e: 
				print(e)
		else:
			min_img_size = 800
			max_img_size = 1024
			num_gpus = 1
			steps_per_epoch = 100
			validation_steps = 30
			if args.num_gpus is not None:
				num_gpus = args.num_gpus
			if args.min_img_size is not None:
				min_img_size = args.min_img_size
			if args.max_img_size is not None:
				max_img_size = args.max_img_size
			if args.steps_per_epoch is not None:
				steps_per_epoch = args.steps_per_epoch
			if args.validation_steps is not None:
				validation_steps = args.validation_steps
			if args.learning_rate is not None:
				parser.error('--learning-rate is only allowed when --type=object-detection')

			maskrcnn_trainer = MaskRcnnTrainer(directory, model="/home/imagemonkey/models/resnet/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
			maskrcnn_trainer.train(labels, min_probability = 0.8, num_gpus=num_gpus, min_image_dimension=min_img_size, max_image_dimension=max_img_size, 
									steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)


	if args.command == "list-labels":
		labels = imagemonkey_api.labels()
		for label in labels:
			print(label)