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
### radtse_dlrecon_test.py
### validate_t2.py
### gen_fastMRI_h5.py
### fastMRI.py

