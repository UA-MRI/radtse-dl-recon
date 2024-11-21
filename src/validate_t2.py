import numpy as np
import os
import torch
from torch.optim.swa_utils import AveragedModel
from models.UnrolledNetwork import RADTSEUnrolledNetwork
from operators.A_functions import *
import dataset.DataGenerator
from glob import glob

import pdb
import matplotlib.pyplot as plt


import yaml

# selecte GPUs
os.system('printenv | grep "CUDA_VISIBLE_DEVICES"')

# Data locations
coils = 6
nlin = 384
SS = 1
    
if nlin == 384:
    key = 'FB_384'
else:
    key = '_320'

h5_dir = '/clusterscratch/tonerbp/data/h5_data/h5_radtse_pcs/' # dir with data
    

# could get this info from the h5 files but hard coding it here
ncol = 512 # nlin, ncol
img_dims = (320,320) # nx, ny
kspace_dims = (nlin, ncol)


shared_params = 0 # end to end training or parameter sharing among cascades
filters = 64 # channels of hidden layers

ncascades = 5 # total cascades in network
nconvolutions = 5 # convolutions per denoiser block

alpha = 0.5

DC_layer = 'DCGD'
kspace_loss = 1
hdr = 1
req = 0
ema = 1
composite_input = 0
disjoint_loss = 0

denoiser = 'CNN'
complex_network = True
corner_penalty = False


odir = f'/clusterscratch/tonerbp/dlrecon_radtse/results/radtse_pcs_{nlin}lin/'
model_name = f'finufft_randfulltraj_{denoiser}_{ncascades}x{nconvolutions}_{filters}f_SS{SS}_kloss{kspace_loss}_{alpha:.0e}alpha/'
# model_name = f'{denoiser}_{ncascades}x{nconvolutions}_{filters}f_SS{SS}_kloss{kspace_loss}_{alpha:.0e}alpha/'


odir = f'{odir}{model_name}'
if not complex_network:
    odir = odir.replace(denoiser, f'{denoiser}REAL')

if ema:
    pretrain_dir = '' # odir
    odir = f'{odir[:-1]}_ema/'
if composite_input:
    pretrain_dir = '' # odir
    odir = f'{odir[:-1]}_compinp/'
if req:
    # pretrain_dir = odir
    odir = f'{odir[:-1]}_req/'
if disjoint_loss:
    odir = f'{odir[:-1]}_djl/'
    
# find old model to load if available
models = glob(f'{odir}models/EPOCH_*.pth')

print(odir)


with open(f'{odir}config.yaml', 'r') as yf:
    config = yaml.safe_load(yf)
config['DC_layer'] = DC_layer
config['max_iter'] = 5
if 'composite_input' not in config:
    config['composite_input'] = 0

nlin = 384
ncol = 512
key = 'FBpaper_384'
test_h5 = f'{h5_dir}radtse_{coils:02d}coils_{key}_valid.h5'
R = 5 / 12
odir = f'{odir}RADTSE_{nlin}_valid_{int(np.round(R*nlin))}_retro/'
dictionary_path = test_h5.replace('/radtse','/SEPGDICT')
roi_path = f'{h5_dir}ROIlabels_{key}_valid.h5'

## Test the model
print('loading test generator')
print(test_h5)
# test set

kspace_dims = (nlin, ncol)

test_generator = dataset.DataGenerator.DataGeneratorRADTSE(test_h5,
                                                           phase='test',
                                                           config=config,
                                                           R=R,
                                                           dcf_method='dcfmax1',
                                                           prescan_norm_power=1)
print('Test samples available:', len(test_generator))

# initialize model
traj, dcf, _ = test_generator.__get_traj_dcf_dict__()
A = RADTSE_ForwardOp(kspace_dims=kspace_dims, img_dims=img_dims, traj=traj, dcf=dcf)
AH = RADTSE_AdjointOp(kspace_dims=kspace_dims, img_dims=img_dims, traj=traj, dcf=dcf)

model = RADTSEUnrolledNetwork(A=A, AH=AH, config=config)
if ema:
    ema_model = AveragedModel(model, avg_fn=get_ema_multi_avg_fn(0.999))
else:
    ema_model = None

"""
for idx in range(len(test_generator)):
    print(idx)
    _,_,_,_ = test_generator.__getitem__(idx)

pdb.set_trace()
"""

for model_path in sorted(models):
    print(model_path)
    
    checkpoint = torch.load(model_path)
    NUM_EPOCHS = len(checkpoint['history']['train_loss'])

    
    if os.path.exists(f'{odir}t2val/t2_error_{NUM_EPOCHS}epochs.csv'):
        print(f'{odir}t2val/t2_error_{NUM_EPOCHS}epochs.csv already found, skipping')
        continue
    
    
    
    # Save TE images from select slices    
    model.validate_t2(test_generator, out_dir=odir,
                     dictionary_path=dictionary_path,save_prefix=f'RADTSE{nlin}PCS',
                      load_model_path=model_path, roi_path=roi_path, ema_model=ema_model)


    # break

