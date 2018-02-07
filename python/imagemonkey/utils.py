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
		self._model_output_dir = training_dir + os.path.sep + "output"
		self._model_output_tmp_dir = self._model_output_dir + os.path.sep + "tmp"
		self._api = API(api_version=1)

		if clear_before_start:
			self.clear_training_dir()
		else:
			if not self.is_training_dir_empty():
				raise ImageMonkeyGeneralError("training directory %s needs to be empty" %(self._training_dir,)) 

		#create output and output tmp dir
		if not os.path.exists(self._model_output_dir):
			os.makedirs(self._model_output_dir)

		if not os.path.exists(self._model_output_tmp_dir):
			os.makedirs(self._model_output_tmp_dir)


	def clear_training_dir(self):
		for f in os.listdir(self._training_dir):
			filePath = os.path.join(self._training_dir, f)
			try:
				if os.path.isfile(filePath):
					os.unlink(filePath)
				elif os.path.isdir(filePath): shutil.rmtree(filePath)
			except Exception as e:
				log.debug("Couldn't clean training directory %s" %(self._training_dir,))
				raise ImageMonkeyGeneralError("Couldn't clean training directory %s" %(self._training_dir,)) 


	def is_training_dir_empty(self):
		if len(os.listdir(self._training_dir)) == 0:
			return True
		return False

	def _create_category_dir(self, category):
		path = self._training_dir + os.path.sep + category
		if os.path.exists(path):
			log.info("Directory %s already exists!" %(path,))
			raise ImageMonkeyGeneralError("Directory %s already exists!" %(path,)) 
		
		os.makedirs(path)

	def _category_dir_exists(self, category):
		path = self._training_dir + os.path.sep + category
		if os.path.exists(path):
			return True
		return False


	def _run_command(self, command):
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
		for line in iter(process.stdout.readline, b''):
			print(">>> " + line.rstrip())


	def _train(self):

		log.debug("Starting tensorflow retrain")
		categories_dir = self._training_dir
		cmd = ("python " + TENSORFLOW_DIR + os.path.sep + "tensorflow" + os.path.sep + "examples" + os.path.sep + 
			"image_retraining" + os.path.sep + "retrain.py" + " --image_dir " + categories_dir + os.path.sep
			+ " --output_graph " + self._model_output_dir + " --output_labels " + self._model_output_dir 
			+ " --intermediate_output_graphs_dir " + self._model_output_tmp_dir + " --model_dir " + self._model_output_tmp_dir
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
				folder = self._training_dir + os.path.sep + validation.label
				self._api.download_image(elem.image.uuid, folder)

		self._train()




