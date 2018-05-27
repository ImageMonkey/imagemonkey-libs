FROM debian:stretch

RUN apt-get update && apt-get install -y git python3 python3-pip neovim dos2unix

#pip 10 has a bug, which makes it fail with: AttributeError: module 'pip' has no attribute 'get_installed_distributions' 
#so we are using 9.0, see https://github.com/pypa/pipenv/issues/1996
RUN pip3 install --upgrade pip==9.0

RUN pip3 install requests
RUN pip3 install tensorflow
RUN pip3 install Pillow

RUN mkdir -p /home/imagemonkey/bin

RUN git clone https://github.com/bbernhard/imagemonkey-libs.git /home/imagemonkey/imagemonkey-libs

RUN git clone https://github.com/tensorflow/models.git tensorflow_models

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN ln -s /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/monkey.py /home/imagemonkey/bin/monkey
RUN chmod +x /home/imagemonkey/bin/monkey
RUN echo "export PATH=\"/home/imagemonkey/bin/:$PATH\"" > ~/.bashrc

