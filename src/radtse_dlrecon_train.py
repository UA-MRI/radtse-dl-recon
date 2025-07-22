import numpy as np
import os
import torch
from torch.optim import NAdam, Adam
from torch.optim.swa_utils import AveragedModel
from models.RADTSE_UnrolledNetwork import RADTSEUnrolledNetwork
from operators.A_functions import *
from models.final_layers import Weighted_L1L2, TE_Weighted_L1L2, TE_PSNR
from dataset.DataGenerator import DataGeneratorRADTSE
from utils.utils import get_ema_multi_avg_fn
from glob import glob
import pdb
from models.train_model import train_model
import yaml
import time

# IMAGE TYPE
image_type = 'composite' # composite or pcs
if image_type == 'composite':
    img_channels = 1 # 1 for composite images
else:
    img_channels = 4 # number or principle componenets for pc images

# SET SEED
np.random.seed(2024)

# GPUs
os.system('printenv | grep "CUDA_VISIBLE_DEVICES"')

# DATA INFO
nlin = 516 # number of radial views in input data
ncol = 512 # number of readout points in input data
img_dims = [256, 256] # this will overwrite what is found in input data

# INPUT DATA LOCATION
h5_dir = '/clusterscratch/tonerbp/data/h5_data/h5_radtse_CAMDTECT/' # dir with data
train_h5 = f'{h5_dir}radtse_CAMDTECT_train.h5'
valid_h5 = f'{h5_dir}radtse_CAMDTECT_valid.h5'

# MODEL PARAMETERS
DC_layer = 'DCGD' # type of DC layer. 'DCGD' or 'DCPM'
shared_params = 0 # 0: end to end training or 1: parameter sharing among cascades
is_residual = 1 # make CNN residual
dropout_p = 0 # dropout probability
filters = 64 # channels of hidden layers
ncascades = 5 # total cascades in network
nconvolutions = 5 # convolutions per denoiser block
denoiser = 'CNN' # CNN currently supported--could import your own architecture
complex_network = True # complex weights or append imag parts into channel dim
hdr = False # high dynamic range loss

# LOSS FUNCTION OPTIONS
alpha = 0.5 # l1 weighting in loss
beta = 1-alpha # l2 weighting in loss
# kspace_loss = 0 for image loss, 1 for kspace loss, 2 for sum of both
if image_type == 'composite':
    kspace_loss = 0 # use image space loss for composite
else:
    kspace_loss = 1 # use kspace loss for pcs
if kspace_loss == 2:
    gamma = 0.5 # kspace weighting when kspace and image loss are summed
else:
    gamma = kspace_loss # kspace weighting when kspace and image loss are summed
gamma_str = f'{gamma}'.replace('.','p')
hdr_eps= 1e-3 # epsilon for HDR loss
corner_penalty = False # penalizes non-zero frequencies in the corners outside of radial trajectory
disjoint_loss = False # 1: calculate loss only on frequencies not used in input, 0: calculate loss on all frequencies
dcf_method = 'ramp' # dcf method--options: 'ramp' 'pipe' or 'ones'

# TRAINING PARAMETERS
NUM_EPOCHS = 100 # number of training epochs
batch_size = 1 # batch size
lr = 1e-4 # learning rate
use_scheduler = 1 # learning rate scheduler
weight_decay= 1e-7 # optimizer weight decay
init_gain = 1e-1 # initialization for CNN weights
dc_init = 1e-1 # initialization for DC weights
ema = 1 # exponential moving average

# SET OUTPUT PATH
odir = f'/clusterscratch/tonerbp/dlrecon_radtse/results/radtse_{image_type}_{nlin}lin/'
model_name = f'{denoiser}_{ncascades}x{nconvolutions}/'
odir = f'{odir}{model_name}'
if not os.path.exists(odir):
    os.makedirs(odir)
print(odir)

# FIND OLD MODEL TO LOAD IF AVAILABLE--CONTINUE TRAINING
pretrain_dir = ''
pretrain_models = glob(f'{pretrain_dir}models/EPOCH_*.pth')
if len(pretrain_models):
    pretrain_path = sorted(pretrain_models)[-1]
else:
    pretrain_path = ''
models = glob(f'{odir}models/EPOCH_*.pth')
if len(models) > 0:
    model_path = sorted(models)[-1]
    print(f'Model path: {model_path}')
else:
    model_path = ''

if not os.path.exists(model_path):
    model_path = None
    if os.path.exists(pretrain_path):
        print('Previous model not found, starting from pretrain:')
        print(pretrain_path)
    else:
        print('Neither previous model nor pretrain found. Starting from scratch.')


# SAVE CONFIGURATIONS
config = {}

config['odir'] = odir
# DATA DETAILS
config['nlin'] = nlin
config['h5_dir'] = h5_dir
config['valid_h5'] = valid_h5
config['train_h5'] = train_h5
config['ncol'] = ncol
config['etl'] = 32
config['img_dims'] = img_dims

# MODEL DETAILS
config['shared_params'] = shared_params
config['is_residual'] = is_residual
config['dropout_p'] = dropout_p
config['filters'] = filters
config['batch_size'] = batch_size
config['ncascades'] = ncascades
config['nconvolutions'] = nconvolutions
config['bias'] = 1
config['stride'] = 1
config['kernel_size'] = 3
config['in_channels'] = img_channels
config['out_channels'] = img_channels
config['GNVN'] = False
config['DC_layer'] = DC_layer
config['max_iter'] = None
config['denoiser_model'] = denoiser
config['complex_network'] = complex_network
config['ema'] = ema
config['disjoint_loss'] = disjoint_loss
config['dcf_method'] = dcf_method

# TRAINING DETAILS
config['normalize'] = True
config['alpha'] = alpha
config['beta'] = beta
config['gamma'] = gamma
config['lr'] = lr
config['weight_decay'] = weight_decay
config['init_gain'] = init_gain
config['init_bias'] = 0
config['dc_init'] = dc_init
config['pretrain'] = os.path.exists(pretrain_path)
if config['pretrain']:
    config['pretrain_path'] = pretrain_path
config['NUM_EPOCHS'] = NUM_EPOCHS
config['load_model_path'] = model_path
config['parallel'] = False
config['optimizer'] = 'Adam'
config['use_scheduler'] = use_scheduler
config['kspace_loss'] = kspace_loss
config['TE_loss'] = 0
config['hdr'] = hdr
config['hdr_eps'] = hdr_eps
config['corner_penalty'] = corner_penalty

# SAVE CONFIGURATIONS
with open(f'{odir}config.yaml', 'w') as yf:
    data = yaml.dump(config, yf)

# CREATE DATA GENERATORS
print('loading valid generator')
print(valid_h5)
# validation set
valid_generator = DataGeneratorRADTSE(valid_h5,
                                      image_type,
                                      phase='valid',
                                      config=config,
                                      dcf_method=dcf_method,
                                      R=0.5)
print('Validation batches to process:', len(valid_generator))
print('loading train generator')
print(train_h5)
# training set
train_generator = DataGeneratorRADTSE(train_h5,
                                      image_type,
                                      phase='train',
                                      config=config,
                                      dcf_method=dcf_method)
print('Training batches to process:', len(train_generator))

# INITIALIZE LOSS FUNCTION
criterion = Weighted_L1L2(alpha=alpha, beta=beta, hdr=hdr, hdr_eps=hdr_eps)
if image_type == 'composite':
    PSNR = None
else:
    PSNR = TE_PSNR(magnitude=True)
    
# INITIALIZE MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
traj, dcf, _ = valid_generator.__get_traj_dcf_dict__()
if image_type == 'composite':
    A = RadialMulticoilForwardOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
    AH = RadialMulticoilAdjointOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
else:
    A = RADTSE_ForwardOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)
    AH = RADTSE_AdjointOp(kspace_dims=(nlin,ncol), img_dims=img_dims, traj=traj, dcf=dcf)

model = RADTSEUnrolledNetwork(A=A, AH=AH, config=config, criterion=criterion)
if ema:
    ema_model = AveragedModel(model, avg_fn=get_ema_multi_avg_fn(0.999))
else:
    ema_model = None

num_params = model.get_num_params()
print(f'{num_params} trainable parameters')

## INITIALIZE OPTIMIZERS
dataloaders = {'train': train_generator, 'valid': valid_generator}
if config['optimizer'] == 'NAdam':
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# LOAD PRETRAINED WEIGHTS IF SELECTED
if os.path.exists(pretrain_path):
    print('Pretrain model found, previous model not found. Starting from pretrain')
    # load old states
    checkpoint = torch.load(pretrain_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if ema:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

# TRAIN MODEL
[model, ema_model] = train_model(model, optimizer, dataloaders, config, A, AH, tensorboard=False,
                                 save_train_curve=True, PSNR=PSNR, ema_model=ema_model)

print(f'Results in {odir}')

