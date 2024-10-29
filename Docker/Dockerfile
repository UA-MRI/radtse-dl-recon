ARG UBUNTU_VERSION=20.04

# Let's first install CUDA
ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.0-base-ubuntu${UBUNTU_VERSION} as base

ARG ARCH
ARG CUDA
ARG CUDNN=8.1.1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

ARG USER_ID=1000005004
ARG GROUP_ID=1000000513

RUN echo $GROUP_ID $USER_ID
RUN addgroup --gid $GROUP_ID user
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN useradd -l -u $USER_ID -g $GROUP_ID user
RUN mkdir /home/user
RUN chown -R user.user /home/user

# The following is to automatically select a locale during installation
RUN export DEBIAN_FRONTEND=noninteractive; \
    export DEBCONF_NONINTERACTIVE_SEEN=true; \
    echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections; \
    echo 'tzdata tzdata/Zones/Etc select US' | debconf-set-selections; \
    apt-get update -qqy \
 && apt-get install -qqy --no-install-recommends \
        tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# install tensorflow dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        #libcudnn8=${CUDNN}+cuda${CUDA} \
        libcudnn8 \
        libcudnn8-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip 

# Install TensorRT if not building for PowerPC
#RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
#        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
#         libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

RUN apt-get update && apt-get install -y \
     nvidia-cuda-toolkit 

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
     python3 \
     python3-pip \
     libexpat1-dev \
     libicu-dev \
     libigraph0-dev

RUN python3 -m pip --no-cache-dir install --upgrade \
     pip \
     setuptools==59.5

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get update && apt-get install -y build-essential autoconf gsl-bin libgsl-dev wget unzip vim git cmake

## xterm
RUN apt-get update && apt-get install -y x11-common x11-xserver-utils
RUN apt-get update && apt-get install -qqy x11-apps
RUN apt-get update && apt-get install -qqy x11-apps
RUN apt-get update && apt-get install -y  xterm


# Packages openEXR needs
RUN apt-get update && apt-get install -y libopenexr-dev


RUN python3 -m pip install --no-cache-dir numpy matplotlib h5py kornia pyyaml

# OPTOX DOWNLOAD AND SETUP--NO INSTALLATION YET
# WORKDIR /tmp
# RUN git clone https://github.com/midas-tum/optox.git
# RUN chown -R user /tmp/optox
# RUN chown -R user /usr/local/lib/python3.8/dist-packages
# RUN chown -R user /usr/local/bin
# ARG CUDA_BIN_PATH=/usr/local/cuda
# USER user
# WORKDIR /tmp/optox

# pybind11
RUN python3 -m pip install --user --no-cache-dir pybind11 sphinx sphinxjp.themes.basicstrap sphinx-rtd-theme breathe


# Pytorch
# RUN python3 -m pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
# RUN python3 -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /tmp
RUN git clone https://github.com/midas-tum/optox.git
RUN chown -R user /tmp/optox
RUN chown -R user /usr/local/lib/python3.8/dist-packages
RUN chown -R user /usr/local/bin
ARG CUDA_BIN_PATH=/usr/local/cuda
USER user
WORKDIR /tmp/optox
RUN python3 install.py --pytorch

# MERLIN
WORKDIR /tmp
RUN git clone https://github.com/midas-tum/merlin.git
WORKDIR /tmp/merlin
RUN python3 -m pip install --user --no-cache-dir merlinth-mri

WORKDIR /tmp

# Pytorch
# RUN python3 -m pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# torchkbnufft
RUN python3 -m pip install --user --no-cache-dir torchkbnufft

# pytorch cufinufft
RUN python3 -m pip install --no-cache-dir cufinufft
RUN python3 -m pip install --user -U git+https://github.com/flatironinstitute/pytorch-finufft.git
