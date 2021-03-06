#!/usr/bin/python3

import argparse
import logging
import sys
import os
import io
import traceback
from tabulate import tabulate
sys.path.insert(1, os.path.join(sys.path[0], ('..' + os.path.sep + "..")))

from pyimagemonkey import API
from pyimagemonkey import Type
from pyimagemonkey import TensorflowTrainer
from pyimagemonkey import MaskRcnnTrainer
from pyimagemonkey import DefaultTrainingStatistics
from pyimagemonkey import LimitDatasetFilter
from pyimagemonkey import OptimalNumOfImagesPerLabelFilter
from pyimagemonkey import TestImageClassificationModel
from pyimagemonkey import TestMaskRcnnModel 
from pyimagemonkey import TensorBoard

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
        logging.basicConfig(level=logging.INFO) 

        parser = argparse.ArgumentParser(prog='PROG')
        parser.add_argument("--verbose", help="verbosity", required=False, default=False)
        subparsers = parser.add_subparsers(help='', dest='command')

        #add subparser for 'train'
        train_parser = subparsers.add_parser('train', help='train your own model')
        train_parser.add_argument('--labels', help='list of ImageMonkey labels that you want to train your model on. default: all labels', default=None, required=False)
        train_parser.add_argument("--delimiter", help="label delimiter", default="|")
        train_parser.add_argument("--type", help="type (object-detection | image-classification | object-segmentation)", required=True)
        train_parser.add_argument("--steps-per-epoch", help="training steps per epoch", required=False, default=None, type=int)
        train_parser.add_argument("--verbose", help="verbosity", required=False, default=False)
        train_parser.add_argument("--epochs", help="num of epochs you want to train", required=False, default=None, type=int)
        train_parser.add_argument("--max-deviation", help="max deviation", required=False, default=None, type=float)
        train_parser.add_argument("--images-per-label", help="number of images per class", required=False, default=None, type=int)
        train_parser.add_argument("--min-probability", help="minimum probability", required=False, default=0.8, type=float)
        train_parser.add_argument("--tensorboard-screenshot", help="create tensorboard screenshot", required=False, default=True, type=bool)
        train_parser.add_argument("--find-optimal-num-of-images-per-label", help="Find optimal number of images per label", required=False, default=False, type=bool)


        #add subparser for 'list-labels'
        list_labels_parser = subparsers.add_parser('list-labels', help='list all labels that are available at ImageMonkey')
        list_labels_parser.add_argument("--verbose", help="verbosity", required=False, default=False)

        list_validations_parser = subparsers.add_parser('list-validations', help='list the validations together with their count')
        list_validations_parser.add_argument("--min-probability", help="minimum probability", required=False, default=0.8, type=float)
        list_validations_parser.add_argument("--min-count", help="minimum count", required=False, default=0, type=int)
        list_validations_parser.add_argument("--verbose", help="verbosity", required=False, default=False)

        list_annotations_parser = subparsers.add_parser('list-annotations', help='list the annotations together with their count')
        list_annotations_parser.add_argument("--min-probability", help="minimum probability", required=False, default=0.8, type=float)
        list_annotations_parser.add_argument("--min-count", help="minimum count", required=False, default=0, type=int)
        list_annotations_parser.add_argument("--verbose", help="verbosity", required=False, default=False)

        test_model_parser = subparsers.add_parser('test-model', help='test your model')
        test_model_parser.add_argument("--type", help="type (object-detection | image-classification | object-segmentation)", required=True)
        test_model_parser.add_argument("--model", help="Path to model file", required=True)
        test_model_parser.add_argument("--labels", help="Path to labels", required=True)
        test_model_parser.add_argument("--image", help="Path to image", required=True)
        test_model_parser.add_argument("--output-image", help="Path to output image", required=False, default=None)
        test_model_parser.add_argument("--verbose", help="verbosity", required=False, default=False)

        args = parser.parse_args()
        if args.verbose:
                logging.basicConfig(level=logging.DEBUG) 


        imagemonkey_api = API(api_version=1)


        if args.command == "train":
                train_type = None
                directory = ""
                tensorboard_logdir = ""
                if args.type == "object-detection":
                        train_type = Type.OBJECT_DETECTION
                        directory = "/tmp/object_detection/"
                        tensorboard_logdir = directory
                elif args.type == "image-classification":
                        train_type = Type.IMAGE_CLASSIFICATION
                        directory = "/tmp/image_classification/"
                        tensorboard_logdir = "/tmp/retrain_logs"
                elif args.type == "object-segmentation":
                        train_type = Type.OBJECT_SEGMENTATION
                        directory = "/tmp/object_segmentation/"
                        tensorboard_logdir = directory
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

                num_images_per_label = None
                if args.images_per_label is not None:
                        num_images_per_label = args.images_per_label

                max_deviation = 0.1
                if args.max_deviation is not None:
                        max_deviation = args.max_deviation

                min_probability = 0.8
                if args.min_probability is not None:
                        min_probability = args.min_probability

                filter_dataset = None
                if args.find_optimal_num_of_images_per_label:
                        filter_dataset = OptimalNumOfImagesPerLabelFilter(imagemonkey_api, labels, min_probability, max_deviation)
                if num_images_per_label is not None:
                        filter_dataset = LimitDatasetFilter(num_images_per_label=num_images_per_label, max_deviation=max_deviation)

                count_annotations = False
                if train_type == Type.OBJECT_DETECTION or train_type == Type.OBJECT_SEGMENTATION:
                        count_annotations = True
                statistics = DefaultTrainingStatistics(count_annotations)
                cmd = ''
                for arg in sys.argv:
                        cmd += ' ' + arg
                statistics.command = cmd 

                if train_type == Type.OBJECT_DETECTION or train_type == Type.IMAGE_CLASSIFICATION:
                        if args.steps_per_epoch is not None:
                                parser.error('--steps-per-epoch is only allowed when --type=object-segmentation')
                        if args.epochs is not None:
                                parser.error('--epochs is only allowed when --type=object-segmentation')

                        if train_type == Type.IMAGE_CLASSIFICATION:
                                statistics.basemodel = "inception_v3"
                        elif train_type == Type.OBJECT_DETECTION:
                                statistics.basemodel = "ssd_mobilenet_v1_coco_11_06_2017"

                        try:
                                tensorflow_trainer = TensorflowTrainer(directory, clear_before_start=True, 
                                                                                                                tf_object_detection_models_path="/root/tensorflow_models/",
                                                                                                                statistics=statistics, filter_dataset=filter_dataset)
                                tensorflow_trainer.train(labels, min_probability=min_probability, train_type=train_type, learning_rate=args.learning_rate)
                        
                                if args.tensorboard_screenshot:
                                    tensorboard = TensorBoard(tensorboard_logdir, "/usr/bin/tensorboard_screenshot.js", tensorflow_trainer.statistics_dir+"/graphs.png")
                                    tensorboard.start()
                                    tensorboard.screenshot()
                                    tensorboard.stop()
                        except Exception as e: 
                                traceback.print_exc()
                                sys.exit(1)
                else:
                        steps_per_epoch = 100
                        validation_steps = 30
                        epochs = 30
                        if args.steps_per_epoch is not None:
                                steps_per_epoch = args.steps_per_epoch
                        if args.epochs is not None:
                                epochs = args.epochs

                        statistics.basemodel = "resnet50"

                        try:
                                maskrcnn_trainer = MaskRcnnTrainer(directory, filter_dataset=filter_dataset, statistics=statistics)
                                maskrcnn_trainer.train(labels, min_probability=min_probability, 
                                                                                steps_per_epoch=steps_per_epoch,epochs=epochs)
                        except Exception as e: 
                                traceback.print_exc()
                                sys.exit(1)


        if args.command == "list-labels":
                labels = imagemonkey_api.labels()
                for label in labels:
                        print(label)

        if args.command == "list-validations" or args.command == "list-annotations":
                min_probability = 0.8
                if args.min_probability is not None:
                        min_probability = args.min_probability

                min_count = 0
                if args.min_count is not None:
                        min_count = args.min_count

                res = None
                if args.command == "list-validations":
                        res = imagemonkey_api.list_validations(min_probability, min_count)
                else:
                        res = imagemonkey_api.list_annotations(min_probability, min_count)
                table = []
                for r in res:
                        table.append([r["label"], r["count"]])
                print(tabulate(table, ["Label", "Count"], tablefmt="grid"))

        if args.command == "test-model":
                if args.type == "image-classification":
                        directory = "/tmp/image_classification_test/"
                        test_image_classification_model = TestImageClassificationModel(args.model, args.labels, directory, clear_before_start=True)
                        test_image_classification_model.label_image(args.image, args.output_image)
                if args.type == "object-segmentation":
                        directory = "/tmp/object_segmentation_test/"
                        test_object_segmentation_model = TestMaskRcnnModel(args.model, args.labels, directory, clear_before_start=True)
                        test_object_segmentation_model.annotate_image(args.image)
