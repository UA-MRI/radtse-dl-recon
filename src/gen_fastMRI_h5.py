import numpy as np
import os
from glob import glob
from dataset import h5_generator_fastMRI

# debug stuff
import pdb
import matplotlib.pyplot as plt
import h5py
import time

# data parameters
accel_type = 'PI'  # simulated undersampling strategy: 'PI' = Parallel Imaging, 'CS' = Compressed Sensing
center = 0.1  # percent of fully sampled central region along ky phase-encoding, e.g. 0.1 := floor(10% * 28) ky center lines = 2 ky center lines
img_dims = (320,320)
kspace_dims = (404, 512) # nlin, ncol


# directories and paths
cl = 'train'
input_h5_dir = f'/clusterscratch/tonerbp/data/fastMRI/multicoil_{cl}/'
input_h5_path_list = glob(f'{input_h5_dir}*.h5')
output_h5_dir = f'/clusterscratch/tonerbp/data/h5_data/h5_fastMRI/multicoil_{cl}/'
if not os.path.exists(output_h5_dir):
    os.makedirs(output_h5_dir)

ct = 0
# h5 generator
slice_list = np.zeros(len(input_h5_path_list))


for idx in range(10): # len(input_h5_path_list)):
    
    input_h5_path = input_h5_path_list[idx]
    
    with h5py.File(input_h5_path, 'r') as h5file:
        
        if h5file['kspace'].shape[0] !=16:
            continue
        
        slice_list[idx] = h5file['kspace'].shape[0]
    print(f'h5 file {idx+1} of {len(input_h5_path_list)}')
    
    for accel in [3]:
        filename = input_h5_path.split('/')[-1]
        out_name = filename.replace('.h5',f'_a{accel:03d}.h5')
        output_h5_path = f'{output_h5_dir}{out_name}'
        
        if os.path.exists(output_h5_path):
            continue
        print(output_h5_path)
        print(f'accleration = {accel}')
    
        
        generator = h5_generator_fastMRI.h5_generator_fastMRI(input_h5_path,
                                                              output_h5_path,
                                                              acceleration=accel,
                                                              accel_type=accel_type,
                                                              center=center,
                                                              traj_type='radial',
                                                              nlin=kspace_dims[0],
                                                              ncol=kspace_dims[1],
                                                              img_dims=img_dims)

        
        generator.h5_generate()

        ct = ct + 1
        print('\n\n')

print(f'ct = {ct}')
print(f'unique numslices = {np.unique(slice_list)}')

# pdb.set_trace()




