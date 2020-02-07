import subprocess
import time
import os
import signal
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class TensorBoard(object):
    def __init__(self, path_to_log_dir, path_to_tensorboard_screenshot, screenshot_output_dir):
        self._path_to_log_dir = path_to_log_dir
        self._path_to_tensorboard_screenshot = path_to_tensorboard_screenshot
        self._screenshot_output_dir = screenshot_output_dir
        self._process = None

        if not os.path.exists(path_to_tensorboard_screenshot):
            raise RuntimeError("tensorboard_screenshot.js doesn't exist at %s" %path_to_tensorboard_screenshot)

    def start(self):
        logging.info("Starting tensorboard...")
        self._process = subprocess.Popen(["tensorboard", "--logdir", self._path_to_log_dir])

    def stop(self):
        log.info("Stopping tensorboard")
        os.kill(self._process.pid, signal.SIGINT)

    def screenshot(self):
        log.info("Waiting 60sec before taking a screenshot...")
        log.info("Take screenshot")
        p = subprocess.Popen(["node", self._path_to_tensorboard_screenshot, "http://127.0.0.1:6006", self._screenshot_output_dir])
        output, err = p.communicate()
        return_code = p.returncode
        if return_code != 0:
            raise RuntimeError("Couldn't create snapshot: %s" %output) 
