import os
import sys
import requests
from PIL import Image
import io

LABEL = ""


BASEURL = 'https://api.imagemonkey.io' #'http://127.0.0.1:8081'

current_dir = os.path.dirname(os.path.abspath(__file__))

def _confirm(files):
	print("Are you sure you want to donate these %d items?\n\n" %(len(files)))
	for file in files:
		print(path + os.path.sep + file)

	print("\n")
	line = input("Are you sure? [yes/no]\n")
	if line == "yes":
		return True
	return False


def _push(filename, label):
	url = BASEURL + '/v1/donate?add_sublabels=false'

	img_bytes = _load_img_and_resize(full_path)

	multipart_formdata = {
		'image': ('file.jpg', img_bytes, 'image/jpg'),

	}

	print("Pushing: %s" %(full_path,))
	try:
		response = requests.post(url, data={'label' : label}, files=multipart_formdata)
		if not response.status_code == 200:
			print("Couldn't push %s: failed with status code: %d, error: %s" %(f, response.status_code, response.text,))
	except Exception as e:
		print("Timeout for %s: " %(f,))

def _load_img_and_resize(path):
	max_dimension = 1024 #px

	img = Image.open(path)

	#get image's size and scale it
	scale_factor = 1.0
	width, height = img.size
	if width > height:
		scale_factor = max_dimension/width
	else:
		scale_factor = max_dimension/height

	if scale_factor > 1.0:
		scale_factor = 1.0

	new_width = round(width * scale_factor)
	new_height = round(height * scale_factor)


	#get binary representation
	img = img.resize((new_width, new_height), Image.ANTIALIAS)

	output = io.BytesIO()
	img.save(output, format='JPEG')
	return output.getvalue()


if __name__ == "__main__":
	if LABEL == "":
		print("Please provide a valid label!")
		sys.exit(1)


	path = current_dir + os.path.sep + LABEL
	if not os.path.exists(path):
		print("Folder '%s' does not exist!" %(path,))
		sys.exit(1)

	
	files = os.listdir(path)

	if not _confirm(files):
		print("aborted due to user input")
		sys.exit(1)

	for f in files:
		full_path = path + os.path.sep + f
		_push(full_path, LABEL)