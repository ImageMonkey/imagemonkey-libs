import subprocess
import sys
import logging
import os
import shutil
import pyimagemonkey.tensorflow_helper as tf_helper
from pyimagemonkey.exceptions import *
import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
import pyimagemonkey.helper as helper 

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class TestImageClassificationModel(object):
        def __init__(self, path_to_model, path_to_labels, output_dir, clear_before_start=False):
                self._path_to_model = path_to_model
                self._path_to_labels = path_to_labels
                self._output_dir = output_dir
                self._output_tmp_dir = output_dir + os.path.sep + "tmp"
                self._output_images_dir = output_dir + os.path.sep + "images"
                self._label_image_py = self._output_tmp_dir + os.path.sep + "label_retrain.py"

                if clear_before_start:
                        clear_output_dir(self._output_dir)

                if not os.path.exists(self._output_tmp_dir):
                        os.makedirs(self._output_tmp_dir)

                if not os.path.exists(self._output_images_dir):
                        os.makedirs(self._output_images_dir)

                if not os.path.exists(self._output_dir):
                        raise ImageMonkeyGeneralError("output directory %s doesn't exist" %(self._output_dir,)) 

                if not os.path.exists(self._label_image_py):
                        tf_helper.download_release_specific_label_image_py("v1.8.0", self._label_image_py)

        def _run_command(self, command, cwd=None, env=None):
                process = None
                if cwd is None:
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stdout, shell=True, universal_newlines=True)
                else:
                        if env is None:
                                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stdout, shell=True, universal_newlines=True, cwd=cwd)
                        else:
                                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stdout, shell=True, universal_newlines=True, cwd=cwd, env=env)

                for line in iter(process.stdout.readline, b''):
                        print(line.rstrip())
                        log.info(line.rstrip())
                        if process.poll() is not None:
                                log.info("Done")
                                return

        def _load_graph(self, model_file):
                graph = tf.Graph()
                graph_def = tf.GraphDef()

                with open(model_file, "rb") as f:
                        graph_def.ParseFromString(f.read())
                with graph.as_default():
                        tf.import_graph_def(graph_def)

                return graph


        def _read_tensor_from_image_file(self, file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
                input_name = "file_reader"
                output_name = "normalized"
                file_reader = tf.read_file(file_name, input_name)
                if file_name.endswith(".png"):
                        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
                elif file_name.endswith(".gif"):
                        image_reader = tf.squeeze(
                        tf.image.decode_gif(file_reader, name="gif_reader"))
                elif file_name.endswith(".bmp"):
                        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
                else:
                        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")

                float_caster = tf.cast(image_reader, tf.float32)
                dims_expander = tf.expand_dims(float_caster, 0)
                resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
                normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
                sess = tf.Session()
                result = sess.run(normalized)

                return result

        def _load_labels(self, label_file):
                label = []
                proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
                for l in proto_as_ascii_lines:
                        label.append(l.rstrip())
                return label

        def label_image(self, path_to_image, output_image=None):
                log.debug("Testing image classification model")
                #cmd = ("python " + self._label_image_py + " --image=" + path_to_image +
                #               " --output_layer=final_result --input_layer=Mul --graph=" + self._path_to_model + " --labels=" + self._path_to_labels)
                #self._run_command(cmd)
                input_height = 299
                input_width = 299
                input_mean = 0
                input_std = 255
                input_layer = "Mul"
                output_layer = "final_result"

                graph = self._load_graph(self._path_to_model)
                t = self._read_tensor_from_image_file(path_to_image, input_height=input_height, input_width=input_width, input_mean=input_mean,
                                                                                                input_std=input_std)

                input_name = "import/" + input_layer
                output_name = "import/" + output_layer
                input_operation = graph.get_operation_by_name(input_name)
                output_operation = graph.get_operation_by_name(output_name)

                with tf.Session(graph=graph) as sess:
                        results = sess.run(output_operation.outputs[0], {
                                input_operation.outputs[0]: t
                        })
                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                labels = self._load_labels(self._path_to_labels)
                for i in top_k:
                        print(labels[i], results[i])

                if output_image is not None:
                        img = cv2.imread(path_to_image, -1)
                        if img is not None:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                bottom_left_corner_of_text = (10,30)
                                font_scale = 1
                                font_color = (255,255,255)
                                line_type = 2

                                for i in top_k:
                                        label_str = str(labels[i]) + ": " + str(results[i])
                                        cv2.putText(img, label_str, bottom_left_corner_of_text, font, font_scale, font_color, line_type)
                                        bottom_left_corner_of_text = (bottom_left_corner_of_text[0], (bottom_left_corner_of_text[1] + 50))
                                cv2.imwrite(output_image, img)



        
