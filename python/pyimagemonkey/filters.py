import copy

from pyimagemonkey.exceptions import *

class DatasetFilter(object):
	def __init__(self):
		pass

	def filter(self, data):
		raise NotImplementedError("process() not implemented")

class OptimalNumOfImagesPerLabelFilter(DatasetFilter):
        def __init__(self, api, labels, min_probability, max_deviation):
                self._api = api
                self._labels = labels
                self._min_probability = min_probability
                self._max_deviation = max_deviation
                super(OptimalNumOfImagesPerLabelFilter, self).__init__()

        def filter(self, data):
                res = self._api.list_validations(self._min_probability, 0)
                label_counts = []
                for r in res:
                        if r["label"] in self._labels:
                                label_counts.append(r["count"])

                min_count = min(int(v) for v in label_counts)

                limitDatasetFilter = LimitDatasetFilter(0, self._max_deviation)
                return limitDatasetFilter.process(data)

class LimitDatasetFilter(DatasetFilter):
	def __init__(self, num_images_per_label, max_deviation):
		super(LimitDatasetFilter, self).__init__()
		self._max_elems_per_label = num_images_per_label
		self._max_deviation = max_deviation

		self._max_deviation_elems = int(float(self._max_deviation) * self._max_elems_per_label)

	def filter(self, data):
		_data = copy.deepcopy(data)

		num_validations = {}
		num_images = {}


		for elem in _data:
			validations = [] 
			for validation in elem.validations:
				num_per_label = 0
				try:
					num_per_label = num_validations[validation.label]
				except KeyError:
					num_per_label = 0

				if num_per_label < self._max_elems_per_label:
					try:
						num_validations[validation.label] += 1
					except KeyError:
						num_validations[validation.label] = 1

					validations.append(validation)

			elem.validations = validations


			annotations = []
			temp_image_labels = {}
			for annotation in elem.annotations:
				num_per_label = 0
				try:
					num_per_label = num_images[annotation.label]
				except KeyError:
					num_per_label = 0
				if num_per_label < self._max_elems_per_label:
					annotations.append(annotation)

					if annotation.label not in temp_image_labels:
						temp_image_labels[annotation.label] = None

			for l in temp_image_labels:
				try:
					num_images[l] += 1
				except KeyError:
					num_images[l] = 1

					

			elem.annotations = annotations

		for n in num_validations:
			if ((num_validations[n] > (self._max_elems_per_label + self._max_deviation_elems)) 
				or (num_validations[n] < (self._max_elems_per_label - self._max_deviation_elems))):
				raise ImageMonkeyDatasetFilterError("number of images for label '%s' doesn't match expected value (expected: %d, got: %d, derivation: %d percent)" 
														%(n, self._max_elems_per_label, num_validations[n], int(self._max_deviation * 100)))

		for n in num_images:
			if ((num_images[n] > (self._max_elems_per_label + self._max_deviation_elems)) 
				or (num_images[n] < (self._max_elems_per_label - self._max_deviation_elems))):
				raise ImageMonkeyDatasetFilterError("number of images for label '%s' doesn't match expected value (expected: %d, got: %d, derivation: %d percent)" 
														%(n, self._max_elems_per_label, num_images[n], int(self._max_deviation * 100)))

		return _data
