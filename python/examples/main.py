import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from imagemonkey import API


if __name__ == "__main__":
	api = API(api_version=1)
	res = api.export(["dog"])
	print res