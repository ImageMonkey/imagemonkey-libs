FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git dos2unix curl nginx nginx-extras wget unzip python3-pip python3-tabulate python-opencv

RUN rm /usr/bin/python \
   && ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /home/imagemonkey/bin

RUN git clone https://github.com/bbernhard/imagemonkey-libs.git /home/imagemonkey/imagemonkey-libs

# the tensorflow developers are changing/deprecating stuff so frequently that we can't use the master branch
# so we checkout an (arbitrary but fixed) specific commit to get a reproducible docker image.
RUN git clone https://github.com/tensorflow/models.git /root/tensorflow_models \
	&& cd /root/tensorflow_models && git checkout 8ffcc2fa3287d031a228860ce574f34c0718cc89 \
	&& cd ~

RUN git clone https://github.com/matterport/Mask_RCNN /home/imagemonkey/mask_rcnn

RUN pip3 install --upgrade setuptools
RUN pip3 install -r /home/imagemonkey/mask_rcnn/requirements.txt
RUN pip3 install --upgrade pip \
	&& hash -r pip3
RUN cd /home/imagemonkey/mask_rcnn/ && python setup.py install

RUN ln -s /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/monkey.py /home/imagemonkey/bin/monkey
RUN chmod +x /home/imagemonkey/bin/monkey
RUN echo "export PATH=\"/home/imagemonkey/bin/:$PATH\"" > ~/.bashrc

RUN mkdir -p /home/imagemonkey/models
RUN mkdir -p /home/imagemonkey/models/resnet/v0.2/
RUN cd /home/imagemonkey/models/resnet/v0.2/ && curl -L https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --output ./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
RUN cd ~/


RUN pip3 install tensorflow

#until this issue is fixed (https://github.com/matterport/Mask_RCNN/issues/566) we pin keras to version 2.0.8
RUN pip3 install --force-reinstall keras==2.0.8


RUN pip3 install --force-reinstall scikit-image==0.13

RUN cd /tmp && wget -O protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN cd /tmp && unzip protoc.zip -d /tmp/protoc
RUN cd /root/tensorflow_models/research/ \
	&& /tmp/protoc/bin/protoc object_detection/protos/*.proto --python_out=. \
	&& rm -rf /notebooks 

WORKDIR /
ENTRYPOINT ["/bin/bash"]

ENV PYTHONPATH $PYTHONPATH:/root/tensorflow_models/research/object_detection/utils

RUN mkdir -p /tmp/image_classification_test/

# docker run -it imagemonkey-train /bin/bash
