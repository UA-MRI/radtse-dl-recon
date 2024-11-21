import numpy as np
import os
import torch
from torch.optim import NAdam
from torch.optim.swa_utils import AveragedModel
from models.RADTSE_UnrolledNetwork import RADTSEUnrolledNetwork
from operators.A_functions import *
from models.final_layers import Weighted_L1L2
from dataset.DataGenerator import DataGeneratorRADTSE
from glob import glob
import pdb
import matplotlib.pyplot as plt
from models.train_model import train_model
import yaml

# IMAGE TYPE
image_type = 'pcs' # composite or pcs
    
# GPUs
os.system('printenv | grep "CUDA_VISIBLE_DEVICES"')

# DATA INFO
coils = 6 # number of channels found in input data (virtual or otherwise)
ncol = 512 # number of readout points in input data
img_dims = [320, 320] # this will overwrite what is fund in test data

# TEST DATA INFOamma
h5_dir = '/clusterscratch/tonerbp/data/h5_data/h5_radtse_pcs/' # dir with data
test_keys = ['FBpaper_384'] # list of h5 file names

# TEST PARAMETERS
vmax = 99 # percentile to scale max to in output images
R =  6 / 12 # 12 # retrospective under-sampling rate
prescan_norm_power = 1 # exponent to raise prescan normalization map to (1 = use as is, 0 = do not use at all)
test_idx = [13] # None # list of slices to reconstruct, or set to None for all slices
q = -1
fov = 256 # fov for output images and error calculation
routine = 'test_model' # test_model or error_metrics
signal_threshold = 0 # whether or not to threshodl the T2 map by anatomical image signal

# TEST PARAMETERS SPECIFIC TO IMAGE TYPE
if image_type == 'composite':
    TEs = None # n/a for composite
    save_t2 = False # n/a for composite
    save_gif = False # n/a for composite
    save_pc = True # for composite, the "pc image" is actually just the composite
    save_h5 = False # save output as .h5 for more analysis
else:
    TEs = [4,10,11,21] # list of TEs to save images of
    save_t2 = True # save T2 images
    save_gif = True # save gif of all TEs for given slice
    save_pc = True # save pc images
    save_h5 = False # save output as .h5 for more analysis

# MODEL PARAMET5RS
ncascades = 5 # total cascades in network
nconvolutions = 5 # convolutions per denoiser block
denoiser = 'CNN' # CNN currently supported--could import your own architecture
train_nlin = 384 # radial views of training data


# FIND TRAINED MODEL
model_dir = f'/clusterscratch/tonerbp/dlrecon_radtse/results/radtse_{image_type}_{train_nlin}lin/'
model_name = f'{denoiser}_{ncascades}x{nconvolutions}/'
model_dir = f'{model_dir}{model_name}'

## MOST RECENT MODEL
models = glob(f'{model_dir}models/EPOCH_*.pth') # list models
model_path = sorted(models)[-1] # most recent model
# model_path = f'{model_dir}models/bestLOSS.pth' # best training loss

print(f'Model path: {model_path}')
if not os.path.exists(model_path):
    print(f'Model {model_path} not found, exiting')
    exit()

# READ CONFIG PARAMS FROM TRAINING
with open(f'{model_dir}config.yaml', 'r') as yf:
    config = yaml.safe_load(yf)

# LOOP THROUGH TEST KEYS
for key in test_keys:
    
    nlin = int(key.split('_')[-1]) # nlin of test
    odir = f'{model_dir}{key}_{int(np.round(nlin*R))}/' # output directory
    test_h5 = f'{h5_dir}radtse_{coils:02d}coils_{key}_test.h5' # input file
    print(test_h5)
    
    if save_h5:
        h5_dir_out = f'{odir}h5data/' # output location for h5 files
    else:
        h5_dir_out = None
    if image_type == 'composite':
        dictionary_path = None
    else:
        dictionary_path = test_h5.replace('/radtse','/SEPGDICT') # input dictionary files
    roi_path = f'{h5_dir}ROIlabels_{key}_test.h5' # input rois
    if not os.path.exists(roi_path) or image_type == 'composite':
        roi_path = None # if set to none, does not calculate ROI means

    kspace_dims = (nlin, ncol)
    
    # BUILD THE TESTING DATA GENERATOR
    print('loading test generator')
    test_generator = DataGeneratorRADTSE(test_h5,
                                         image_type,
                                         phase='test',
                                         config=config,
                                         R=R,
                                         prescan_norm_power=prescan_norm_power)
    print('Test samples available:', len(test_generator))
    
    # INITIALIZE MODEL
    traj, dcf, _ = test_generator.__get_traj_dcf_dict__() # traj and dcf for forward and adjoint models
    if image_type == 'composite':
        A = RadialMulticoilForwardOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
        AH = RadialMulticoilAdjointOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
    else:
        A = RADTSE_ForwardOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
        AH = RADTSE_AdjointOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)

    model = RADTSEUnrolledNetwork(A=A, AH=AH, config=config)
    if config['ema']:
        ema_model = AveragedModel(model, avg_fn=get_ema_multi_avg_fn(0.999))
    else:
        ema_model = None


    if routine == 'error_metrics':
        model.error_metrics(test_generator, indices=test_idx,out_dir=odir, save_prefix=f'{key}_{int(np.round(nlin*R))}_mag',
                              load_model_path=model_path, ema_model=ema_model, mag=True, TEs=TEs, q=q, vmax=vmax, fov=fov)
    else:
        # SAVE RESULTS
        model.test_model(test_generator, indices=test_idx, out_dir=odir, TEs=TEs, save_t2=save_t2, save_gif=save_gif,
                         dictionary_path=dictionary_path,save_prefix=f'RADTSE{nlin}{image_type}',save_pc=save_pc,
                         load_model_path=model_path, vmax=vmax, h5_dir=h5_dir_out, roi_path=roi_path,
                         ema_model=ema_model, fov=fov, signal_threshold=signal_threshold)
        
               
print(f'Results in {odir}')



