FROM debian:stretch

RUN apt-get update && apt-get install -y git python3 python3-pip neovim dos2unix curl

#pip 10 has a bug, which makes it fail with: AttributeError: module 'pip' has no attribute 'get_installed_distributions' 
#so we are using 9.0, see https://github.com/pypa/pipenv/issues/1996
RUN pip3 install --upgrade pip==9.0

RUN pip3 install requests
RUN pip3 install tensorflow
RUN pip3 install Pillow

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /home/imagemonkey/bin

RUN git clone https://github.com/bbernhard/imagemonkey-libs.git /home/imagemonkey/imagemonkey-libs

RUN git clone https://github.com/tensorflow/models.git tensorflow_models

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



# TODO REMOVE ME AGAIN
#COPY pyimagemonkey/* /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/
#COPY pyimagemonkey/scripts/monkey.py /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/
#RUN dos2unix /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/monkey.py
#RUN mkdir -p /tmp2
#RUN rm /home/imagemonkey/bin/monkey
#RUN ln -s /home/imagemonkey/imagemonkey-libs/python/pyimagemonkey/scripts/monkey.py /home/imagemonkey/bin/monkey
#RUN chmod +x /home/imagemonkey/bin/monkey

