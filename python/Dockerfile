FROM debian:stretch

RUN apt-get update && apt-get install -y git python3 python3-pip

#pip 10 has a bug, which makes it fail with: AttributeError: module 'pip' has no attribute 'get_installed_distributions' 
#so we are using 9.0, see https://github.com/pypa/pipenv/issues/1996
RUN pip3 install --upgrade pip==9.0

RUN pip3 install requests
RUN pip3 install tensorflow
RUN pip3 install Pillow

RUN git clone https://github.com/bbernhard/imagemonkey-libs.git

RUN git clone https://github.com/tensorflow/models.git tensorflow_models

COPY pyimagemonkey/scripts /imagemonkey-libs/python/pyimagemonkey/scripts
