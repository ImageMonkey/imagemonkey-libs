import json
import datetime
import sys

class TrainingStatistics(object):
	def __init__(self, output_path = None):
		self._output_path = output_path
		self._output = None

	def generate(self, data):
		raise NotImplementedError("generate not implemented")

	@property
	def output_path(self):
		return self._output_path

	@output_path.setter
	def output_path(self, p):
		self._output_path = p

	def save(self):
		if (self._output is not None) and (self._output_path is not None):
			with open(self._output_path, 'w') as f:
				f.write(self._output)


class DefaultTrainingStatistics(TrainingStatistics):
	def __init__(self, count_annotations=False):
		super(DefaultTrainingStatistics, self).__init__()
		self._command = None
		self._basemodel = None
		self._count_annotations = count_annotations
		self._class_names = None

	@property
	def command(self):
		return self._command

	@command.setter
	def command(self, command):
		self._command = command

	@property
	def basemodel(self):
		return self._basemodel

	@basemodel.setter
	def basemodel(self, basemodel):
		self._basemodel = basemodel

	@property
	def class_names(self):
		return self._class_names

	@class_names.setter
	def class_names(self, class_names):
		self._class_names = class_names

	def generate(self, data):
		output = {}
		cnt = {}
		image_cnt = {}
		labels = {}

		for elem in data:
			if self._count_annotations:
				temp_image_labels = {}
				for annotation in elem.annotations:
					try:
						cnt[annotation.label]["num"] += 1
					except KeyError:
						cnt[annotation.label] = {"num": 1}

					if annotation.label not in labels:
						labels[annotation.label] = None

					if annotation.label not in temp_image_labels:
						temp_image_labels[annotation.label] = None

				for l in temp_image_labels:
					try:
						image_cnt[l]["num"] += 1
					except KeyError:
						image_cnt[l] = {"num" : 1}

			else:
				temp_image_labels = {}
				for validation in elem.validations:
					try:
						cnt[validation.label]["num"] += 1
					except KeyError:
						cnt[validation.label] = {"num": 1}

					if validation.label not in labels:
						labels[validation.label] = None

					if validation.label not in temp_image_labels:
						temp_image_labels[validation.label] = None

				for l in temp_image_labels:
					try:
						image_cnt[l]["num"] += 1
					except KeyError:
						image_cnt[l] = {"num" : 1}

		if self._count_annotations:
			output["training"] = {"images": image_cnt, "annotations": cnt, "command": self._command, "based_on": self._basemodel}
		else:
			output["training"] = {"images": image_cnt, "validations": cnt, "command": self._command, "based_on": self._basemodel}

		if self._class_names is not None:
			output["training"]["class_names"] = self._class_names

		now = datetime.datetime.now()
		output["created"] = now.strftime("%Y-%m-%d %H:%M")
		output["trained_on"] = list(labels.keys())

		self._output = json.dumps(output)

