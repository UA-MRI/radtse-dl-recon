import numpy as np
import os
import torch
from torch.optim import Adam, NAdam
from torch.optim.swa_utils import AveragedModel
from models.UnrolledNetwork import UnrolledNetwork
from operators.A_functions import *
from dataset.fastMRI_DataGenerator import *
from glob import glob

import pdb
import matplotlib.pyplot as plt

from models.train_model import train_model
from models.final_layers import Weighted_L1L2

import yaml
import h5py

# set seed
np.random.seed(2024)

# Data locations
h5_dir = '/clusterscratch/tonerbp/data/h5_data/h5_fastMRI/' # dir with data
h5_files = glob(f'{h5_dir}multicoil*/*a003.h5') # list all available preprocessed h5 files

slices_per_h5 = 16
ntest = 1 # number of h5 files to save for testing
test_h5_list = h5_files[:ntest] # list of test h5 files
ntrainvalid = len(h5_files) - ntest # total h5 files used for train and valid
pct_train = 0.9 # percent of data to use as training vs validation
ntrain = int(np.floor(pct_train*ntrainvalid)) # how many files for training
train_h5_list = h5_files[ntest:ntrain+ntest] # training data
valid_h5_list = h5_files[ntest+ntrain:] # validation data


# could get this info from the h5 files but hard coding it here
nlin = 384
ncol = 512
kspace_dims = (nlin,ncol) # nlin, ncol
img_dims = (320,320) # nx, ny

# train and test parameters
NUM_EPOCHS = 5 # number of training epochs
test_idx = [8, 24] # slices to test
shared_params = 0 # end to end training or parameter sharing among cascades
is_residual = 1 # make CNN residual
dropout_p = 0 # dropout probability
filters = 64 # channels of hidden layers
batch_size = 1 # batch size -- must be factor of slices_per_h5
denoiser = 'CNN' # CNN currently supported--could import your own architecture

ncascades = 5 # total cascades in network
nconvolutions = 5 # convolutions per denoiser block

alpha = 0.5
beta = 1-alpha

lr = 1e-4
weight_decay= 1e-7
init_gain = 1e-1

use_scheduler = 1
DC_layer = 'DCGD'
kspace_loss = 0
dcf_loss = 0
max_iter = 1
ema = 1
disjoint_loss = 0


pretrain_dir = ''
odir = f'/clusterscratch/tonerbp/dlrecon_radtse/results/fastMRI_comp/{denoiser}_3x/'


if not os.path.exists(odir):
    os.makedirs(odir)
    
# find old model to load if available
pretrain_path = f'{pretrain_dir}models/bestLOSS.pth'
models = glob(f'{odir}models/EPOCH_*.pth')
if len(models) == 0:
    model_path = f'{odir}models/LATEST.pth'
else:
    model_path = sorted(models)[-1]
    
print(f'Model path: {model_path}')
if not os.path.exists(model_path):
    model_path = None
    if os.path.exists(pretrain_path):
        print('Previous model not found, starting from pretrain')
        print(f'Pretrain path: {pretrain_path}')
    else:
        print('Neither previous model nor pretrain found. Starting from scratch.')

print('loading valid generator')
# validation set
valid_generator = GeneratorFASTMRI(valid_h5_list, kspace_dims,
                                                 batch_size=batch_size,
                                                 h5_slices=slices_per_h5,
                                                 shuffle=True,
                                                 test=False)
print('loading train generator')
# training set
train_generator = GeneratorFASTMRI(train_h5_list, kspace_dims,
                                                 batch_size=batch_size,
                                                 h5_slices=slices_per_h5,
                                                 shuffle=True,
                                                 test=False)


print('Training batches to process:', len(train_generator))
print('Validation batches to process:', len(valid_generator))


# save configuration
config = {}

config['odir'] = odir
# data details
# config['coils'] = coils
# config['norm'] = norm
config['nlin'] = nlin
config['h5_dir'] = h5_dir
config['valid_h5'] = valid_h5_list
config['train_h5'] = train_h5_list
config['test_h5'] = test_h5_list
config['ncol'] = ncol

# model details
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
config['in_channels'] = 1
config['out_channels'] = 1
config['GNVN'] = False
config['DC_layer'] = DC_layer
config['max_iter'] = max_iter
config['denoiser_model'] = denoiser
config['complex_network'] = 1
config['composite_input'] = 0
config['ema'] = ema
config['corner_penalty'] = 0
config['disjoint_loss'] = disjoint_loss

# training details
config['alpha'] = alpha
config['beta'] = beta
config['lr'] = lr
config['weight_decay'] = weight_decay
config['init_gain'] = init_gain
config['init_bias'] = 0
config['dc_init'] = 0.1
config['pretrain'] = os.path.exists(pretrain_path)
if config['pretrain']:
    config['pretrain_path'] = pretrain_path
config['NUM_EPOCHS'] = NUM_EPOCHS
config['load_model_path'] = model_path
config['parallel'] = False
config['optimizer'] = 'Adam'
config['use_scheduler'] = use_scheduler
config['kspace_loss'] = kspace_loss
config['dcf_loss'] = dcf_loss

with open(f'{odir}config.yaml', 'w') as yf:
    data = yaml.dump(config, yf)
    

# initialize model, move to gpu if available
traj, dcf = train_generator.__get_traj_dcf__(ncol, nlin)
A = RadialMulticoilForwardOp(kspace_dims, img_dims, traj, dcf)
AH = RadialMulticoilAdjointOp(kspace_dims, img_dims, traj, dcf)

criterion = Weighted_L1L2(alpha=alpha, beta=beta)
model = UnrolledNetwork(A=A, AH=AH, config=config, criterion=criterion)

if ema:
    ema_model = AveragedModel(model, avg_fn=get_ema_multi_avg_fn(0.999))
else:
    ema_model = None
    
# print model overview
# print(model)
num_params = model.get_num_params()
print(f'{num_params} trainable parameters')


## Train the model
dataloaders = {'train': train_generator, 'valid': valid_generator}
if config['optimizer'] == 'NAdam':
    optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model = train_model(model, optimizer, dataloaders, config, A, AH, tensorboard=False,
                    save_train_curve=True, ema_model=ema_model)



## Test the model
print('loading test generator')
# test set
test_generator = GeneratorFASTMRI(test_h5_list,
                                                batch_size=1,
                                                h5_slices=slices_per_h5,
                                                shuffle=False,
                                                test=True)
print(test_h5_list)
print('Test samples available:', len(test_generator))

model_path = f'{odir}models/bestLOSS.pth'
# model.gen_metrics(test_generator, out_dir=odir, save_prefix=f'fastMRI',
#                   load_model_path=model_path)

model.test_model(test_generator, indices=test_idx,
                 save_prefix=f'fastMRI', load_model_path=model_path)


print(f'Results in {odir}')


