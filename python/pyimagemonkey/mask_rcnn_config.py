from mrcnn.config import Config


class ImageMonkeyConfig(Config):

    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "imagemonkey"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    #IMAGE_MIN_DIM = 256 #800
    #IMAGE_MAX_DIM = 320 #1024

    def __init__(self, num_classes, num_gpus, min_image_dimension, max_image_dimension, 
                    steps_per_epoch, validation_steps):
        #set NUM_CLASSES before calling base class
        #otherwise it won't work
        self.NUM_CLASSES = num_classes + 1 #(num of classes +1 for background)
        self.GPU_COUNT = num_gpus
        self.IMAGE_MIN_DIM = min_image_dimension
        self.IMAGE_MAX_DIM = max_image_dimension
        self.STEPS_PER_EPOCH = steps_per_epoch #Number of training steps per epoch
        self.VALIDATION_STEPS = validation_steps #Number of validation steps
        super().__init__() 
