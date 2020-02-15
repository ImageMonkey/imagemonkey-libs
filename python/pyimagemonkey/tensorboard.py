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
        self._started = False

        if not os.path.exists(path_to_tensorboard_screenshot):
            raise RuntimeError("tensorboard_screenshot.js doesn't exist at %s" %path_to_tensorboard_screenshot)

    def start(self):
        logging.info("Starting TensorBoard...")
        self._process = subprocess.Popen(["tensorboard", "--logdir", self._path_to_log_dir], 
                                            stdout=subprocess.PIPE, universal_newlines=True, stderr=subprocess.STDOUT)
        for line in iter(self._process.stdout.readline, ''):
            logging.info("Waiting for TensorBoard to start...")
            if "TensorBoard" in line: 
                logging.info("TensorBoard started")
                self._started = True
                break
            time.sleep(1)
        

    def stop(self):
        logging.info("Stopping TensorBoard")
        os.kill(self._process.pid, signal.SIGINT)

    def screenshot(self):
        if self._started:
            logging.info("Take Screenshot")
            p = subprocess.Popen(["node", self._path_to_tensorboard_screenshot, "http://127.0.0.1:6006", self._screenshot_output_dir])
            output, err = p.communicate()
            return_code = p.returncode
            if return_code != 0:
                raise RuntimeError("Couldn't create snapshot: %s" %output)
        else:
            logging.info("Cannot take screenshot, as TensorBoard isn't started") 
