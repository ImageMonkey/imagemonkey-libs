import json

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


class ImageClassificationTrainingStatistics(TrainingStatistics):
	def __init__(self):
		super(ImageClassificationTrainingStatistics, self).__init__()

	def generate(self, data):
		output = {}
		
		for elem in data:
			for validation in elem.validations:
				try:
					output[validation.label]["images"]["num"] += 1
				except KeyError:
					output[validation.label] = {"images": {"num": 1}}
		self._output = json.dumps(output)

