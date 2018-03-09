import os
from imagemonkey.exceptions import *
from imagemonkey.api import *
import logging
import shutil
import subprocess
import shlex
import logging
import pip
import sys


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class TensorflowTrainer(object):
	def __init__(self, training_dir, clear_before_start=False, auto_download_tensorflow_train_script = True):
		self._training_dir = training_dir
		self._auto_download_tensorflow_train_script = auto_download_tensorflow_train_script
		self._images_dir = training_dir + os.path.sep + "images"
		self._model_output_dir = training_dir + os.path.sep + "output"
		self._models_dir = training_dir + os.path.sep + "models"
		self._model_output_tmp_dir = self._model_output_dir + os.path.sep + "tmp"
		self._retrain_py_dir = self._models_dir
		self._retrain_py = self._retrain_py_dir + os.path.sep + "retrain.py"
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

		if self._auto_download_tensorflow_train_script:
			if not self._retrain_py_exists():
				installed_tensorflow_version = self._get_installed_tensorflow_version() 
				if installed_tensorflow_version is None:
					raise ImageMonkeyGeneralError("trying to download tensorflow retrain script...couldn't find tenorflow. is it installed?")
				self._download_release_specific_retrain_py(("v" + installed_tensorflow_version))

	def _get_installed_tensorflow_version(self):
		installed_packages = pip.get_installed_distributions()
		for i in installed_packages:
			if i.key == "tensorflow":
				return i.version
		return None

	def _get_available_tensorflow_releases(self):
		releases = []

		url = "https://api.github.com/repos/tensorflow/tensorflow/releases"
		resp = requests.get(url)
		if resp.status_code != 200:
			raise ImageMonkeyGeneralError("Couldn't fetch available tensorflow releases")
		data = resp.json()
		for elem in data:
			releases.append(elem["tag_name"])
		return releases

	def _get_commit_hash_for_tensorflow_release(self, release):
		url = "https://api.github.com/repos/tensorflow/tensorflow/tags"
		resp = requests.get(url)
		if resp.status_code != 200:
			raise ImageMonkeyGeneralError("Couldn't fetch available tensorflow tags")
		data = resp.json()
		for elem in data:
			if elem["name"] == release:
				commit_id = elem["commit"]["sha"] 
				return commit_id
		return None

	def _get_release_specific_retrain_py_url(self, release):
		return 


	def _download_release_specific_retrain_py(self, release):
		releases = self._get_available_tensorflow_releases()
		if release not in releases:
			raise ImageMonkeyGeneralError("'%s' is not a valid tensorflow release" %(release,))
		commit_id = self._get_commit_hash_for_tensorflow_release(release)
		if commit_id is None:
			raise ImageMonkeyGeneralError("fetching tensorflow commit hash...'%s' is not a valid tensorflow release" %(release,))

		#download release specific retrain.py
		url = "https://raw.githubusercontent.com/tensorflow/tensorflow/%s/tensorflow/examples/image_retraining/retrain.py" %(commit_id,)
		resp = requests.get(url)
		if resp.status_code != 200:
			raise ImageMonkeyGeneralError("Couldn't get release specific tensorflow retrain.py")
		with open(self._retrain_py, "wb") as f:
			f.write(resp.content)



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

	def _retrain_py_exists(self):
		if os.path.exists(self._retrain_py):
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
		cmd = ("python " + self._retrain_py + " --image_dir " + self._images_dir + os.path.sep
			+ " --output_graph " + (self._model_output_dir + os.path.sep + "graph.pb") + " --output_labels " + (self._model_output_dir + os.path.sep + "labels.txt")
			+ " --intermediate_output_graphs_dir " + self._model_output_tmp_dir + " --model_dir " + self._models_dir
			+ " --bottleneck_dir " + self._model_output_tmp_dir)
		self._run_command(cmd)

	def train(self, labels, min_probability = 0.8):
		for label in labels:
			if not self._category_dir_exists(label):
				self._create_category_dir(label)

		res = self._api.export(labels, min_probability)
		for elem in res:
			for validation in elem.validations:
				folder = self._images_dir + os.path.sep + validation.label
				self._api.download_image(elem.image.uuid, folder)

		self._train()




