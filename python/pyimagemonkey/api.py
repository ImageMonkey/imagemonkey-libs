import requests
import os
from pyimagemonkey.exceptions import *
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

def _rotate_point(point, angle, center_point=PolyPoint(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point.x - center_point.x, point.y - center_point.y)
    new_point = (new_point.x * cos(angle_rad) - new_point.y * sin(angle_rad),
                 new_point.x * sin(angle_rad) + new_point.y * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point.x + center_point.x, new_point.y + center_point.y)
    return new_point

def _rotate_polygon(polygon, angle, center_point=PolyPoint(0, 0)):
    """Rotates the given polygon which consists of corners represented as (x,y)
    around center_point (origin by default)
    Rotation is counter-clockwise
    Angle is in degrees
    """
    rotated_polygon = []
    for p in polygon:
        rotated_p = rotate_point(p, angle, center_point)
        rotated_polygon.append(rotated_p)
    return rotated_polygon

class Ellipse(object):
	def __init__(self, left, top, rx, ry):
		self._left = left
		self._top = top
		self._rx = rx
		self._ry = ry
		self._angle = 0

	@property
	def rx(self):
		return self._rx

	@property
	def ry(self):
		return self._ry

	@property
	def left(self):
		return self._left

	@property
	def top(self):
		return self._top


	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, angle):
		self._angle = angle

class Rectangle(object):
	def __init__(self, top, left, width, height):
		self._top = top
		self._left = left
		self._width = width
		self._height = height
		self._angle = 0

		self._points = []
		self._points.append(PolyPoint(left, top))
		self._points.append(PolyPoint(left + width, top))
		self._points.append(PolyPoint(left, top + height))
		self._points.append(PolyPoint(left + width, top + height))

	@property
	def top(self):
		return self._top
	@property
	def left(self):
		return self._left
	@property
	def width(self):
		return self._width
	@property
	def height(self):
		return self._height

	def scaled(self, rect):
		if not isinstance(rect, Rectangle):
			raise ValueError("expected rectangle")

		top = self._top
		if self._top > rect.top:
			top = rect.top
		if top < 0:
			top = 0

		left = self._left
		if self._left > rect.left:
			left = rect.left
		if left < 0:
			left = 0

		width = self._width
		if self._width > rect.width:
			width = rect.width

		height = self._height
		if self._height > rect.height:
			height = rect.height

		return Rectangle(top, left, width, height)

	@property
	def center(self):
		return PolyPoint((left + (width/2)), (top + (height/2)))

	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, angle):
		self._angle = angle

	@property
	def scaled_points(self):
		if self._angle != 0:
			return _rotate_polygon(self._points, self._angle, self.center)
		return self._points

	@property
	def points(self):			
		return self._points





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

	#returns the bounding rectangle
	@property
	def rect(self):
		if len(self._points) == 0:
			raise ValueError("Can't compute bounding box of empty list")
		minx, miny = float("inf"), float("inf")
		maxx, maxy = float("-inf"), float("-inf")
		for point in self._points:
			# Set min coords
			if point.x < minx:
				minx = point.x
			if point.y < miny:
				miny = point.y
            # Set max coords
			if point.x > maxx:
				maxx = point.x
			elif point.y > maxy:
				maxy = point.y

		return Rectangle(left=minx, top=miny, width=(maxx - minx), height=(maxy - miny))

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
		self._path = None
		self._folder = None

	@property
	def path(self):
		return self._path

	@path.setter
	def path(self, path):
		self._path = path

	@property
	def folder(self):
		return self._folder

	@folder.setter
	def folder(self, folder):
		self._folder = folder

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

	@property
	def data(self):
		return self._data




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


def _parse_result(data, min_probability):
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
				if "angle" in raw_annotation:
					polygon.angle = raw_annotation["angle"]
				annotation = Annotation(raw_annotation["label"], polygon)
				annotations.append(annotation)
			if(raw_annotation["type"] == "rect"):
				rect = Rectangle(raw_annotation["top"], raw_annotation["left"], raw_annotation["width"], raw_annotation["height"])
				if "angle" in raw_annotation:
					rect.angle = raw_annotation["angle"]
				annotation = Annotation(raw_annotation["label"], rect)
				annotations.append(annotation)
			if(raw_annotation["type"] == "ellipse"):
				ellipse = Ellipse(raw_annotation["left"], raw_annotation["top"], raw_annotation["rx"], raw_annotation["ry"])
				if "angle" in raw_annotation:
					ellipse.angle = raw_annotation["angle"]
				annotation = Annotation(raw_annotation["label"], ellipse)
				annotations.append(annotation)

		for raw_validation in raw_validations:
			validation_probability = 0
			try:
				validation_probability = float(raw_validation["num_yes"]) / (raw_validation["num_yes"] + raw_validation["num_no"])
			except ZeroDivisionError:
				pass
				
			if validation_probability < min_probability:
				log.debug("skipping validation entry, as probability < min. probability")
				continue
			validations.append(Validation(raw_validation["label"], raw_validation["num_yes"], raw_validation["num_no"]))

		res.append(DataEntry(image, validations, annotations))

	return res





class API(object):
	def __init__(self, api_version):
		self._api_version = api_version
		self._base_url = "https://api.imagemonkey.io/"



	def labels(self, show_accessors=False):
		url = None
		if show_accessors:
			url =  self._base_url + "v" + str(self._api_version) + "/label/accessors"
		else:
			url =  self._base_url + "v" + str(self._api_version) + "/label"
		r = requests.get(url)
		if(r.status_code == 500):
			raise InternalImageMonkeyAPIError("Could not perform operation, please try again later")
		elif(r.status_code != 200):
			raise ImageMonkeyAPIError(r["error"])

		data = r.json()
		return data

	def export(self, labels, min_probability = 0.8):
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
		return _parse_result(data, min_probability)

	def download_image(self, uuid, folder, extension=".jpg"):
		if not os.path.isdir(folder):
			raise ImageMonkeyGeneralError("folder %s doesn't exist" %(folder,))

		filename = folder + os.path.sep + uuid + extension
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







