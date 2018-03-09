# Python #

This library is a small wrapper around the [ImageMonkey](https://imagemonkey.io) API. 

**WARNING** The library is still in an alpha stage, which means that the API may change as the development continues. 

## Requirements ##

* Python 3.x is required

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

* Model (re-)training with Tensorflow

Downloads all images from ImageMonkey that are tagged with the label `dog` or `cat` and feeds them directly into Tensorflow to train a new layer on top of a pre-trained image model. The downloaded images are stored in an `images` folder within the training directory (`C:\training`). In case the `clear_before_start` parameter is `True` the whole images directory gets cleared and the images get re-fetched from ImageMonkey every time the script is run. 

Internally the `TensorflowTrainer` class uses the tensorflow [`retrain.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py) script. Usually you do not need to download this script manually, as the ImageMonkey library will take care about that. The file will be automatically downloaded and put into the `models` folder within the specified `training` directory. In case you want to download the file manually, set `auto_download_tensorflow_train_script` to `False` and copy the file into the appropriate folder. 


```python
import logging

from imagemonkey import API
from imagemonkey import TensorflowTrainer

if __name__ == "__main__":
	logging.basicConfig()

	tensorflow_trainer = TensorflowTrainer("C:\\training", clear_before_start=True, auto_download_tensorflow_train_script=True)
	tensorflow_trainer.train(["dog", "cat"], min_probability = 0.8)
```
