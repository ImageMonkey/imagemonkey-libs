import os
import sys

from setuptools import setup
setup(
  name = 'pymonkey',
  python_requires='>=3.5.0',
  packages = ['pymonkey'],
  version = '0.1.16',
  description = 'This library is a small wrapper around the ImageMonkey API.',
  long_description='pymonkey is a small Python library which makes using the ImageMonkey API easy as pie.',
  author = 'Bernhard',
  author_email = 'schluchti@gmail.com',
  url = 'https://github.com/bbernhard/imagemonkey-libs',
  download_url = '', 
  keywords = ['machine learning', 'imagemonkey'],
  classifiers = [],
  install_requires = ['tensorflow>=1.0', 'requests'],
)