import requests
from pyimagemonkey.exceptions import *
from pyimagemonkey.type import *
import pip


def get_installed_tensorflow_version():
	installed_packages = pip.get_installed_distributions()
	for i in installed_packages:
		if i.key == "tensorflow":
			return i.version
	return None


def get_available_tensorflow_releases(product_type=ProductType.TENSORFLOW):
	releases = []
	url = ""
	if product_type == ProductType.TENSORFLOW:
		url = "https://api.github.com/repos/tensorflow/tensorflow/releases"
	elif product_type == ProductType.TENSORFLOW_MODELS:
		url = "https://api.github.com/repos/tensorflow/models/releases"
	else:
		raise ImageMonkeyGeneralError("Invalid Tensorflow Product type")

	resp = requests.get(url)
	if resp.status_code != 200:
		raise ImageMonkeyGeneralError("Couldn't fetch available tensorflow releases")
	data = resp.json()
	for elem in data:
		releases.append(elem["tag_name"])
	return releases

def get_commit_hash_for_tensorflow_release(release, product_type=ProductType.TENSORFLOW):
	url = ""
	if product_type == ProductType.TENSORFLOW:	
		url = "https://api.github.com/repos/tensorflow/tensorflow/tags"
	elif product_type == ProductType.TENSORFLOW_MODELS:
		url = "https://api.github.com/repos/tensorflow/models/tags"
	else:
		raise ImageMonkeyGeneralError("Invalid Tensorflow Product type")
	resp = requests.get(url)
	if resp.status_code != 200:
		raise ImageMonkeyGeneralError("Couldn't fetch available tensorflow tags")
	data = resp.json()
	for elem in data:
		if elem["name"] == release:
			commit_id = elem["commit"]["sha"] 
			return commit_id
	return None

def download_release_specific_retrain_py(release, download_path):
	releases = get_available_tensorflow_releases(product_type=ProductType.TENSORFLOW)
	if release not in releases:
		raise ImageMonkeyGeneralError("'%s' is not a valid tensorflow release" %(release,))
	commit_id = get_commit_hash_for_tensorflow_release(release, product_type=ProductType.TENSORFLOW)
	if commit_id is None:
		raise ImageMonkeyGeneralError("fetching tensorflow commit hash...'%s' is not a valid tensorflow release" %(release,))

	#download release specific retrain.py
	url = "https://raw.githubusercontent.com/tensorflow/tensorflow/%s/tensorflow/examples/image_retraining/retrain.py" %(commit_id,)
	resp = requests.get(url)
	if resp.status_code != 200:
		raise ImageMonkeyGeneralError("Couldn't get release specific tensorflow retrain.py")
	with open(download_path, "wb") as f:
		f.write(resp.content)

def download_release_specific_label_image_py(release, download_path):
	releases = get_available_tensorflow_releases(product_type=ProductType.TENSORFLOW)
	if release not in releases:
		raise ImageMonkeyGeneralError("'%s' is not a valid tensorflow release" %(release,))
	commit_id = get_commit_hash_for_tensorflow_release(release, product_type=ProductType.TENSORFLOW)
	if commit_id is None:
		raise ImageMonkeyGeneralError("fetching tensorflow commit hash...'%s' is not a valid tensorflow release" %(release,))

	#download release specific retrain.py
	url = "https://raw.githubusercontent.com/tensorflow/tensorflow/%s/tensorflow/examples/label_image/label_image.py" %(commit_id,)
	resp = requests.get(url)
	if resp.status_code != 200:
		raise ImageMonkeyGeneralError("Couldn't get release specific tensorflow label_image.py")
	with open(download_path, "wb") as f:
		f.write(resp.content)