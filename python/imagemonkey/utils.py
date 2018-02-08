import os
from imagemonkey.exceptions import *
from imagemonkey.api import *
import logging
import shutil
import subprocess
import shlex
import logging

#TODO: get rid of that!
TENSORFLOW_DIR = "C:\\imagemonkey-libs\\python\\tensorflow"


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class TensorflowTrainer(object):
	def __init__(self, training_dir, clear_before_start=False):
		self._training_dir = training_dir
		self._images_dir = training_dir + os.path.sep + "images"
		self._model_output_dir = training_dir + os.path.sep + "output"
		self._models_dir = training_dir + os.path.sep + "models"
		self._model_output_tmp_dir = self._model_output_dir + os.path.sep + "tmp"
		self._api = API(api_version=1)

		if clear_before_start:
			if os.path.exists(self._images_dir):
				self.clear_images_dir()
		else:
			if not self.is_images_dir_empty():
				raise ImageMonkeyGeneralError("training directory %s needs to be empty" %(self._images_dir,)) 

		#create output, output tmp dir and images dir
		if not os.path.exists(self._model_output_dir):
			os.makedirs(self._model_output_dir)

		if not os.path.exists(self._model_output_tmp_dir):
			os.makedirs(self._model_output_tmp_dir)

		if not os.path.exists(self._images_dir):
			os.makedirs(self._images_dir)

		if not os.path.exists(self._models_dir):
			os.makedirs(self._models_dir)


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


	def is_images_dir_empty(self):
		if len(os.listdir(self._images_dir)) == 0:
			return True
		return False

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


	def _run_command(self, command):
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
		for line in iter(process.stderr.readline, b''):
			print(line.rstrip())
			log.info(line.rstrip())

		#for line in iter(process.stdout.readline, b''):
		#	print(line.rstrip())
		#	log.info(line.rstrip())


	def _train(self):

		log.debug("Starting tensorflow retrain")
		cmd = ("python " + TENSORFLOW_DIR + os.path.sep + "tensorflow" + os.path.sep + "examples" + os.path.sep + 
			"image_retraining" + os.path.sep + "retrain.py" + " --image_dir " + self._images_dir + os.path.sep
			+ " --output_graph " + self._model_output_dir + " --output_labels " + self._model_output_dir 
			+ " --intermediate_output_graphs_dir " + self._model_output_tmp_dir + " --model_dir " + self._models_dir
			+ " --bottleneck_dir " + self._model_output_tmp_dir)
		print(cmd)
		self._run_command(cmd)

	def train(self, labels):
		for label in labels:
			if not self._category_dir_exists(label):
				self._create_category_dir(label)

		res = self._api.export(labels)
		for elem in res:
			for validation in elem.validations:
				folder = self._images_dir + os.path.sep + validation.label
				self._api.download_image(elem.image.uuid, folder)

		self._train()




