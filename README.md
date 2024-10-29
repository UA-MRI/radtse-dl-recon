# dcdl

Data consistent deep learning


## Docker

In the Docker folder you will find everything you need (hopefully) to
build the Docker container that can run these experiments. 

Docker/build_docker will build a container according to the
instructions in Docker/Dockerfile. 

Docker/run_docker.sh will launch the container. 

It is currently set up to mount the folder that contains all this code
on the U of A server. This will need to be accounted for non
Altbach/Bilgin lab people. 

## merlin_src

This contains my experiments written with merlinth, the pytorch
version of merlin. 

## preprocess_src

This directory contains the scripts that have been used to help
produce the datasets. 
