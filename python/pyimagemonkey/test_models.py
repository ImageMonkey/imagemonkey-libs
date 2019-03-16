import subprocess
import sys
import logging
import os
import shutil
from pyimagemonkey.exceptions import *

class TestImageClassificationModel(object):
	def __init__(self, path_to_model, path_to_labels, output_dir, clear_before_start=False):
		self._path_to_model = path_to_model
		self._path_to_labels = path_to_labels
		self._output_dir = output_dir
		self._output_tmp_dir = output_dir + os.path.sep + "tmp"
		self._label_image_py = self._output_tmp_dir + os.path.sep + "label_retrain.py"

		if clear_before_start:
			self.clear_output_dir()

		if not os.path.exists(self._output_tmp_dir):
			os.makedirs(self._output_tmp_dir)

		if not os.path.exists(self._output_dir):
			raise ImageMonkeyGeneralError("output directory %s doesn't exist" %(self._output_dir,)) 

		if not os.path.exists(self._label_image_py):
			tf_helper.download_release_specific_retrain_py("v1.8.0", self._label_image_py)


	def clear_output_dir(self):
		for f in os.listdir(self._output_dir):
			filePath = os.path.join(self._output_dir, f)
			try:
				if os.path.isfile(filePath):
					os.unlink(filePath)
				elif os.path.isdir(filePath): shutil.rmtree(filePath)
			except Exception as e:
				log.error("Couldn't clear output directory %s" %(self._output_dir,))
				raise ImageMonkeyGeneralError("Couldn't clear output directory %s" %(self._output_dir,))



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

	def label_image(self, path_to_image):
		log.debug("Testing image classification model")
		cmd = ("python " + self._label_image_py + " --image=" + path_to_image +
				" --output_layer=final_result --input_layer=Placeholder --graph=" + path_to_model + " --labels=" + self._path_to_labels)
		self._run_command(cmd)

	