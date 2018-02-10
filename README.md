# Python #
## Examples ##

* download images

download all images that are tagged with the label `dog` and store them in `C:\dogs`. We are only interested in images where at least 80% of the people think, that the image is correctly labeled. (`min_probability = 0.8`)

```python
import logging
from imagemonkey import API


if __name__ == "__main__":
	logging.basicConfig()

	api = API(api_version=1)
	res = api.export(["dog"], min_probability = 0.8)

	ctr = 1
	for elem in res:
		print "[%d/%d] Downloading image %s" %(ctr, len(res), elem.image.uuid)
		api.download_image(elem.image.uuid, "C:\\dogs")
		ctr += 1
```
