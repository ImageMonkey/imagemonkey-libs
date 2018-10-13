FROM debian:stretch

RUN apt-get update && apt-get install -y git python3 python3-pip neovim dos2unix curl nginx nginx-extras wget unzip

#pip 10 has a bug, which makes it fail with: AttributeError: module 'pip' has no attribute 'get_installed_distributions' 
#so we are using 9.0, see https://github.com/pypa/pipenv/issues/1996
RUN pip3 install --upgrade pip==9.0 \
	&& pip3 install requests \
    && pip3 install tensorflow \
    && pip3 install Pillow

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /home/imagemonkey/bin

RUN git clone https://github.com/bbernhard/imagemonkey-libs.git /home/imagemonkey/imagemonkey-libs

# the tensorflow developers are changing/deprecating stuff so frequently that we can't use the master branch
# so we checkout an (arbitrary but fixed) specific commit to get a reproducible docker image.
RUN git clone https://github.com/tensorflow/models.git tensorflow_models \
	&& cd tensorflow_models && git checkout 8ffcc2fa3287d031a228860ce574f34c0718cc89 \
	&& cd ~

RUN git clone https://github.com/matterport/Mask_RCNN /home/imagemonkey/mask_rcnn

RUN pip3 install -r /home/imagemonkey/mask_rcnn/requirements.txt
RUN cd /home/imagemonkey/mask_rcnn/ && python3 setup.py install

RUN ln -s /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/monkey.py /home/imagemonkey/bin/monkey
RUN chmod +x /home/imagemonkey/bin/monkey
RUN echo "export PATH=\"/home/imagemonkey/bin/:$PATH\"" > ~/.bashrc

RUN mkdir -p /home/imagemonkey/models
RUN mkdir -p /home/imagemonkey/models/resnet/v0.2/
RUN cd /home/imagemonkey/models/resnet/v0.2/ && curl -L https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --output ./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
RUN cd ~/

#until this issue is fixed (https://github.com/matterport/Mask_RCNN/issues/566) we pin keras to version 2.0.8
RUN pip3 install --force-reinstall keras==2.0.8


RUN pip3 install --force-reinstall scikit-image==0.13

# pin tensorflow to version 1.7.1, newer versions have tensorhub, which we do not support right now, see: 
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining
RUN pip3 install --force-reinstall tensorflow==1.7.1

RUN cd /tmp && wget -O protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN cd /tmp && unzip protoc.zip -d /tmp/protoc
RUN cd /tensorflow_models/research/ \
	&& /tmp/protoc/bin/protoc object_detection/protos/*.proto --python_out=. 

ENV PYTHONPATH $PYTHONPATH:/tensorflow_models/research/object_detection/utils


#RUN export PYTHONPATH=$PYTHONPATH:/tensorflow_models/research/object_detection/utils
