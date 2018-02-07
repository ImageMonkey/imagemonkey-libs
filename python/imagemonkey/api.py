import requests
import os
from imagemonkey.exceptions import *
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class PolyPoint(object):
	def __init__(self, x, y):
		self._x = x
		self._y = y

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y

class Polygon(object):
	def __init__(self, poly_points):
		self._angle = 0
		self._points = []
		for point in poly_points:
			p = PolyPoint(point["x"], point["y"])
			self._points.append(p)

	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, angle):
		self._angle = angle

	@property
	def points(self):
		return self._points

class Validation(object):
	def __init__(self, label, num_yes, num_no):
		self._num_yes = num_yes
		self._num_no = num_no
		self._label = label

	@property
	def label(self):
		return self._label

	@property
	def num_no(self):
		return self._num_no

	@property
	def num_yes(self):
		return self._num_yes


class Image(object):
	def __init__(self, uuid, width, height):
		self._uuid = uuid
		self._width = width
		self._height = height

	@property
	def uuid(self):
		return self._uuid

	@property
	def width(self):
		return self._width

	@property
	def height(self):
		return self._height

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


class Annotation(object):
	def __init__(self, label, data):
		self._data = data
		self._label = label

	@property
	def label(self):
		return self._label




class DataEntry(object):
	def __init__(self, image, validations, annotations):
		self._image = image
		self._annotations = annotations
		self._validations = validations

	@property
	def validations(self):
		return self._validations

	@property
	def annotations(self):
		return self._annotations

	@property
	def image(self):
		return self._image

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


def _parse_result(data):
	res = []
	for elem in data:
		image = Image(elem["uuid"], elem["width"], elem["height"])
		raw_annotations = elem["annotations"]
		raw_validations = elem["validations"]
		annotations = []
		validations = []
		for raw_annotation in raw_annotations:
			if(raw_annotation["type"] == "polygon"):
				polygon = Polygon(raw_annotation["points"])
				polygon.angle = raw_annotation["angle"]
				annotation = Annotation(raw_annotation["label"], polygon)
				annotations.append(annotation)
		for raw_validation in raw_validations:
			validations.append(Validation(raw_validation["label"], raw_validation["num_yes"], raw_validation["num_no"]))

		res.append(DataEntry(image, validations, annotations))

	return res





class API(object):
	def __init__(self, api_version):
		self._api_version = api_version
		self._base_url = "https://api.imagemonkey.io/"



	def labels(self):
		url =  self._base_url + "v" + str(self._api_version) + "/label"
		r = requests.get(url)
		if(r.status_code == 500):
			raise InternalImageMonkeyAPIError("Could not perform operation, please try again later")
		elif(r.status_code != 200):
			raise ImageMonkeyAPIError(r["error"])

		data = r.json()
		return data

	def export(self, labels):
		query = ""
		for x in range(len(labels)):
			query += labels[x]
			if(x != (len(labels) - 1)):
				query += "|"
		url = self._base_url + "v" + str(self._api_version) + "/export"
		params = {"query": query}
		r = requests.get(url, params=params)
		if(r.status_code == 500):
			raise InternalImageMonkeyAPIError("Could not perform operation, please try again later")
		elif(r.status_code != 200):
			data = r.json()
			raise ImageMonkeyAPIError(data["error"])

		data = r.json()
		return _parse_result(data)



	"""def download_images(self, labels, folder):
		url = self._base_url + "v" + str(self._api_version) + "/donation/"
		data = self.export(labels)
		for elem in data:
			img_url = url + data["uuid"]
			log.info("Downloading image %s" % (img_url,))
			r = requests.get(img_url)
			if(r.status_code != 200):
				pass"""

	def download_image(self, uuid, folder):
		if not os.path.isdir(folder):
			raise ImageMonkeyGeneralError("folder %s doesn't exist" %(folder,))

		filename = folder + os.path.sep + uuid
		if os.path.exists(filename):
			raise ImageMonkeyGeneralError("image %s already exists in folder %s" %(uuid,folder))

		url = self._base_url + "v" + str(self._api_version) + "/donation/" + uuid
		response = requests.get(url)
		if response.status_code == 200:
			log.info("Downloading image %s" %(uuid,))
			with open(filename, 'wb') as f:
				f.write(response.content)
		else:
			raise ImageMonkeyAPIError("couldn't download image %s" %(uuid,))







