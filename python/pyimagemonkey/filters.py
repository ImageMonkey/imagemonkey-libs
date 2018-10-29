import copy

from pyimagemonkey.exceptions import *

class DatasetFilter(object):
	def __init__(self):
		pass

	def filter(self, data):
		raise NotImplementedError("process() not implemented")

class LimitDatasetFilter(DatasetFilter):
	def __init__(self, num_images_per_label, max_deviation):
		super(LimitDatasetFilter, self).__init__()
		self._max_elems_per_label = num_images_per_label
		self._max_deviation = max_deviation

		self._max_deviation_elems = int(float(self._max_deviation) * self._max_elems_per_label)

	def filter(self, data):
		_data = copy.deepcopy(data)

		num = {}

		for elem in _data:
			validations = [] 
			for validation in elem.validations:
				num_per_label = 0
				try:
					num_per_label = num[validation.label]
				except KeyError:
					num_per_label = 0

				if num_per_label < self._max_elems_per_label:
					try:
						num[validation.label] += 1
					except KeyError:
						num[validation.label] = 1

					validations.append(validation)

			elem.validations = validations

		for n in num:
			if ((num[n] > (self._max_elems_per_label + self._max_deviation_elems)) 
				or (num[n] < (self._max_elems_per_label - self._max_deviation_elems))):
				raise ImageMonkeyDatasetFilterError("number of images for label '%s' doesn't match expected value (expected: %d, got: %d, derivation: %d percent)" 
														%(n, self._max_elems_per_label, num[n], int(self._max_deviation * 100)))

		return _data