import logging
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from imagemonkey import API
from imagemonkey import TensorflowTrainer

if __name__ == "__main__":
	logging.basicConfig()

	tensorflow_trainer = TensorflowTrainer("C:\\Users\\Bernhard\\imagemonkey_dogs\\training", clear_before_start=True)
	tensorflow_trainer.train(["orange", "glass"])