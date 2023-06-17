FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get install -y git wget cmake
RUN apt-get install -y qt5-qmake

RUN apt install -y python3.8 python3.8-distutils python3-pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

ENV LD_LIBRARY_PATH "/usr/local/lib:$LD_LIBRARY_PATH"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install --upgrade pip

WORKDIR /first-breaks-picking
COPY requirements/basic.txt /first-breaks-picking/basic.txt
RUN pip install -r /first-breaks-picking/basic.txt

COPY . /first-breaks-picking

RUN pip install .

