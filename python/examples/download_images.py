import logging
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from imagemonkey import API


if __name__ == "__main__":
	logging.basicConfig()

	api = API(api_version=1)
	res = api.export(["dog"])

	ctr = 1
	for elem in res:
		print "[%d/%d] Downloading image %s" %(ctr, len(res), elem.image.uuid)
		api.download_image(elem.image.uuid, "C:\\Users\\Bernhard\\imagemonkey_dogs")
		ctr += 1