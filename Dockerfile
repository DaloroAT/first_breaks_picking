FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
ENV QT_QPA_PLATFORM=offscreen

RUN apt-get -y update
RUN apt-get install -y git wget cmake
RUN apt-get install -y git libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx libglib2.0-0 libgtk2.0 qt5-qmake

RUN apt install -y python3.8 python3.8-distutils python3-pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

ENV LD_LIBRARY_PATH "/usr/local/lib:$LD_LIBRARY_PATH"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install --upgrade pip

WORKDIR /first-breaks-picking
COPY pyproject.toml .
RUN pip install .[dependencies]

COPY . /first-breaks-picking
RUN pip install -e .

