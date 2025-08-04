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
This script creates and trains the proposed DL model for RADTSE reconstruction. It requires training data in .h5 format. The dataset requires the following:
- kspace (ncoils, etl, nshots, nsteps, nslice, 2): raw data
- smaps (ncoils, nx, ny, nslice, 2): coil sensitivity maps
- traj (etl, nshots, nsteps, nslice, 2): kspace trajectory coordinates
- dcf (etl, nshots, nsteps, nslice, 2): density compensation function
- dict (npcs, etl, nslice): temporal compression operator (used for TE/T2 network only)

With the following dimensions:
- ncoils: number of (virtual) coils. We compress our data to 6 virtual coils to save memory and ensure all datasets have the same number of coils
- etl: echo train length, or the number of readouts following an excite pulse
- nshots: number of echo trains (number of excite pulses)
- nsteps: number of readout points (2x oversampling in the readout dimension gives 2x the prescribed matrix size)
- nslice: number of slices in the dataset
- nx, ny: image dimensions of the sensitivity maps. These are automatically cropped if larger than the desired image size
- npcs: number of principal components used in temporal compression (used for TE/T2 network only)

In the version of matlab we used to generate the .h5 files (2020a), writing complex datasets was not supported, so the final dimension for complex fields denotes the real and imaginary components of the data.
h5py does support writing and reading complex datasets, which is much faster than reading the real data and converting to complex. So the first time we encounter the .h5 file in python we re-write it as a complex dataset to be read in future iterations. This step could easily be removed if you start with complex datasets (in the `src/dataset/DataGenertor.py` file). 

### radtse_dlrecon_test.py
This script evaluates the trained model. It requires testing data in .h5 format. Optionally, include a T2 dictionary to generate T2 maps and an roi file to do T2 quantification.
The dictionary .h5 file generally has the following fields (although for specific uses not all may be necessary):
- u (etl x npc): compression operator
- s (etl x 1): singular values from SVD compression
- magnetization (npc x natoms): signal decay curves
- normalization (1 x natoms): factors used to normalize magnetization
- lookup_table (natoms x 2): lookup table for (B1, T2)
- TE: echo spacing (in s)
- ETL: etl
- FAdeg: refocusing flip angle (in degrees)
- T1: T1 value used (in s)
- fcoherence (etl x 1): example signal with infinite T1 and T2, used only in calculating effective TE for VFA data
- alpha (etl x 1): Actual refocusing flip angles played by the sequence (radians)

### gen_fastMRI_h5.py
Without real RADTSE data, it is possible to use this package on simulated radial data. This script will create simulated radial data from fastMRI data. 

### fastMRI.py
This script creates, trains, and evaluates a model trained on simulated fastMRI data. 

