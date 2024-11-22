# RADTSE DL RECON

Cascaded unrolled neural network for RADTSE reconstruction


## Docker

In the Docker folder you will find everything you need (hopefully) to
build the Docker container that can run these experiments. 

Docker/build_docker will build a container according to the
instructions in Docker/Dockerfile. 

Docker/run_docker.sh will launch the container. 

It is currently set up to mount the folder that contains all this code
on the U of A server. This will need to be accounted for non
Altbach/Bilgin lab people. 

## src

### radtse_dlrecon_train.py
This script creates and trains the proposed DL model for RADTSE reconstruction. It requires training data in .h5 format

### radtse_dlrecon_test.py
This script evaluates the trained model. It requires testing data in .h5 format. Optionally, include a T2 dictionary and roi file, both in .h5 format, to do T2 quantification.

### gen_fastMRI_h5.py
Without real RADTSE data, it is possible to use this package on simulated radial data. This script will create simulated radial data from fastMRI data. 

### fastMRI.py
This script creates, trains, and evaluates a model trained on simulated fastMRI data. 

