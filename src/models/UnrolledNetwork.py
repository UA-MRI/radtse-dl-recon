import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn

import merlinth
from merlinth.layers.complex_act import ModReLU
from merlinth.losses.pairwise_loss import psnr

from PIL import Image

from models.final_layers import *
from utils.utils import *

from time import time

import h5py

# define a basic CNN block 
def CNN_block(in_channels, out_channels, filters, num_layers, kernel_size, stride, bias, dropout_p=0):
    cnn_layer_list = [] # initialize list of layers
    # first convolution
    cnn_layer_list += [merlinth.layers.ComplexConv2d(in_channels=in_channels,
                                                     out_channels=filters,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding='same',
                                                     bias=bias)]
    # first dropout
    cnn_layer_list += [ComplexDropout(p=dropout_p)]
    # first activation
    cnn_layer_list += [ModReLU(num_parameters=filters)]
    # middle (hidden) layers
    for didx in range(num_layers-2): 
        cnn_layer_list += [merlinth.layers.ComplexConv2d(in_channels=filters,
                                                         out_channels=filters,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding='same',
                                                         bias=bias)]
        cnn_layer_list += [ComplexDropout(p=dropout_p)]
        cnn_layer_list += [ModReLU(num_parameters=filters)]

    # final layer--no activation, no dropout
    cnn_layer_list += [merlinth.layers.ComplexConv2d(in_channels=filters,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding='same',
                                                     bias=True)]
        
    denoiser_block = nn.Sequential(*cnn_layer_list) # denoising block

    return denoiser_block
    
# Define unrolled reconstruction model
class UnrolledNetwork(nn.Module):        
    # initialize the required layers
    def __init__(self, A, AH, config, DC_layer=merlinth.layers.DCGD, criterion=None):
        super().__init__()

        self.config = config # config parameters

        self.A = A # forward model
        self.AH = AH # adjoint model

        self.is_residual = config['is_residual'] # use residual layer or not
        self.T = 1 if config['shared_params'] else config['ncascades']  # shared denoiser network or (new) cascaded denoisers
        self.dropout_p = config['dropout_p'] # percent dropout of feature map

        self.T_end = config['ncascades']  # number of cascades
        self.ND = config['nconvolutions'] # depth of network (number of convolutions)

        self.complex_network = config['complex_network'] # complex weights vs splitting real and imag components in channels

        # loss function
        if criterion is None:
            self.criterion = Weighted_L1L2(alpha=0.5, beta=0.5)
        else:
            self.criterion = criterion

        # set DC layer method
        if config['DC_layer'] == 'DCPM':
            DC_layer = merlinth.layers.DCPM # proximal method
        else:
            DC_layer = merlinth.layers.DCGD # gradient descent
        max_iter = config['max_iter'] # is only used for DCPM

        in_channels = config['in_channels'] # channel dimension of input
        out_channels = config['out_channels'] # channel dimension of output
        self.ncha = in_channels

        # weight initialization for conv layers
        self.init_gain = config['init_gain']
        self.init_bias = config['init_bias']
        # weight initialization for DC layers
        dc_init = config['dc_init']
        
        # convolution parameters
        bias = config['bias']
        kernel_size = config['kernel_size']
        stride = config['stride']
        filters = config['filters']
        
        # get the CNN denoising block
        denoiser_list = []
        for cidx in range(self.T):
            if config['denoiser_model'].lower() == 'didn':
                if self.complex_network:
                    from models.DIDN import DIDN
                    denoiser_list += [DIDN(in_channels, out_channels, filters)]
                else:
                    from models.DIDN_real import DIDN
                    denoiser_list += [DIDN(in_channels*2, out_channels*2, filters)]
            else:
                denoiser_list += [CNN_block(in_channels, out_channels, filters, self.ND,
                                            kernel_size, stride, bias, self.dropout_p)]
        self.denoiser = nn.ModuleList(denoiser_list) # denoising network

        # add data consistency layer(s)
        self.dc = nn.ModuleList([DC_layer(A, AH, weight_init=dc_init, max_iter=max_iter)
                                 for _ in range(self.T)])  # data consistency blocks

        # initialize weights
        if self.init_gain is not None:
            self.apply(self._init_weights)

    # build the model in the forward path
    def forward(self, inputs):

        x = inputs[0]  # undersampled image
        for i in range(self.T_end):  # unrolled network
            ii = i % self.T
            if self.is_residual: # if residual hold onto input to add later
                res = torch.clone(x)
            if not self.complex_network:# if not complex, concatenate real and imaginary parts in channel dim
                x = torch.cat([x.real, x.imag], dim=1)
                
            x = self.denoiser[ii](x)  # denoising regularizer
            
            if not self.complex_network: # if not complex, add real and imaginary back together
                x = x[:,0:self.ncha,:,:] + 1j*x[:,self.ncha:,:,:]
            if self.is_residual:
                x += res #inputs[0] # residual layer

                
            if isinstance(self.dc[ii], merlinth.layers.DCPM):
                for idx in range(1, len(inputs)):
                    if inputs[idx].shape[0] != x.shape[0]:
                        inputs[idx] = torch.cat(x.shape[0]*[inputs[idx].unsqueeze(0)])

            x = self.dc[ii]([x, ] + list(inputs[1:]))  # data consistency

        return x

    def get_num_params(self,trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, merlinth.layers.ComplexConv2d):
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_gain)
            module.bias.data.fill_(self.init_bias)
            

    # test with trained model
    def test_model(self, test_generator, load_model_path=None, indices=None, save_prefix='',
                   save_tiff=False, NUM_EPOCHS=-1):

        out_dir = self.config['odir']
        
        # mode model to device
        print(f'\nTEST USING GPU: {torch.cuda.is_available()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # load old model if path is provided
        if load_model_path is not None:
            # load old states
            checkpoint = torch.load(load_model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            NUM_EPOCHS = len(checkpoint['history']['train_loss'])
        save_prefix = f'{save_prefix}_{NUM_EPOCHS}epochs'
        
        self.eval() # turn dropout off by putting model in train mode    

        # determine which samples from the generator to test
        if indices is None:
            indices = range(len(test_generator)) # do all if not declared
        # loop through slices
        for testidx in indices:
            kspace, smaps = test_generator.__getitem__(testidx)
            traj, dcf = test_generator.__get_traj_dcf__()
            kspace = kspace.to(device)
            smaps = smaps.to(device)
            traj = traj.to(device)
            dcf = dcf.to(device)
            print(f'Testing image {testidx} of {len(test_generator)}')

            # generate input mask
            [nslc, _, nlin, ncol] = kspace.shape
            [_, _, nx, ny] = smaps.shape
            mask = torch.ones([nslc, nlin, ncol]).to(device)
            if config['SS'] >= 2: # random mask keeping TRs intact
                nshots = int(config['nlin'] / config['etl']) # number of TRs
                N = int(nshots // 2) # 2x under-sampling for validation
            else:
                lin_mask = np.zeros(config['nlin'], 1)
                lin_mask[::2,:] = 1
                
                SS_mask[:,lin_mask,:] = 1 # mask for input to network
            noisy = self.AH(kspace, SS_mask, smaps, traj, dcf)
            inputs = [noisy, kspace, SS_mask, smaps, traj, dcf]

            with torch.set_grad_enabled(False):
                pred = self(inputs).cpu().detach() # make prediction
            # move back to cpu for plotting
            for idx,inp in enumerate(inputs):
                inputs[idx] = inp.cpu()

            icase = 0  # display the first example
            for cha in range(self.ncha):
                noisy_mag = np.transpose(inputs[0][icase,cha].abs().squeeze().numpy(), (1,0))
                recon_mag = np.transpose(pred[icase,cha].abs().squeeze().numpy(), (1,0))
                target_mag = np.transpose(targets[icase,cha].abs().squeeze().numpy(), (1,0))
                noisy_phase = np.transpose(inputs[0][icase,cha].angle().squeeze().numpy(), (1,0))
                recon_phase = np.transpose(pred[icase,cha].angle().squeeze().numpy(), (1,0))
                target_phase = np.transpose(targets[icase,cha].angle().squeeze().numpy(), (1,0))

                error_mag = np.abs(recon_mag - target_mag)
                error_phase = np.abs(recon_phase - target_phase)
                error_max = np.percentile(error_mag, 100)
            
                # display the predicted output
                plt.figure()
                plt.subplot(2,4,1)
                plt.imshow(noisy_mag, cmap='gray', vmax=np.percentile(noisy_mag,100))
                plt.title('Magnitude - Noisy',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,2)
                plt.imshow(recon_mag, cmap='gray', vmax=np.percentile(recon_mag,100))
                plt.title('Magnitude - Recon',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,3)
                plt.imshow(target_mag, cmap='gray', vmax=np.percentile(target_mag,100))
                plt.title('Magnitude - Target',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,4)
                plt.imshow(error_mag, vmax=error_max)
                plt.title('Magnitude - Absolute Error',fontsize=8)
                plt.colorbar()
                plt.axis('off')
                plt.subplot(2,4,5)
                plt.imshow(noisy_phase, vmin=-np.pi, vmax=np.pi)
                plt.title('Phase - Noisy',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,6)
                plt.imshow(recon_phase, vmin=-np.pi, vmax=np.pi)
                plt.title('Phase - Recon',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,7)
                plt.imshow(target_phase, vmin=-np.pi, vmax=np.pi)
                plt.title('Phase - Target',fontsize=8)
                plt.axis('off')
                plt.subplot(2,4,8)
                plt.imshow(error_phase)
                plt.title('Phase - Absolute Error',fontsize=8)
                plt.colorbar()
                plt.axis('off')

                # save
                if not os.path.exists(f'{out_dir}images'):
                    os.makedirs(f'{out_dir}images')
                plt.savefig(f'{out_dir}images/{save_prefix}_{testidx:03d}_cha{cha:02d}.png')
                plt.clf()
                plt.close()

                if save_tiff:
                    plt.imshow(error_mag, vmax=error_max, cmap='jet')
                    plt.colorbar()
                    plt.axis('off')
                    plt.savefig(f'{out_dir}images/{save_prefix}_error_{testidx:03d}_cha{cha:02d}.png', bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    
                    noisy_mag = Image.fromarray(noisy_mag/np.percentile(noisy_mag,100), mode='F')
                    noisy_mag.save(f'{out_dir}images/{save_prefix}_noisymag_{testidx:03d}_cha{cha:02d}.tiff','TIFF')
                
                    recon_mag = Image.fromarray(recon_mag/np.percentile(recon_mag,100), mode='F')
                    recon_mag.save(f'{out_dir}images/{save_prefix}_reconmag_{testidx:03d}_cha{cha:02d}.tiff','TIFF')
                
                    target_mag = Image.fromarray(target_mag/np.percentile(target_mag,100), mode='F')
                    target_mag.save(f'{out_dir}images/{save_prefix}_targetmag_{testidx:03d}_cha{cha:02d}.tiff','TIFF')


    
    # test with trained model
    def gen_metrics(self, test_generator, out_dir, indices=None, save_prefix='',
                    load_model_path=None, NUM_EPOCHS=-1):
        # mode model to device
        print(f'\nTEST USING GPU: {torch.cuda.is_available()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # load old model if path is provided
        if load_model_path is not None:
            # load old states
            checkpoint = torch.load(load_model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            NUM_EPOCHS = len(checkpoint['history']['train_loss'])
        save_prefix = f'{save_prefix}_{NUM_EPOCHS}epochs'
        
        self.eval() # turn dropout off by putting model in train mode    

        # determine which samples from the generator to test
        if indices is None:
            indices = range(len(test_generator)) # do all if not declared

        # initialize metrics
        mean_l1 = 0
        mean_l2 = 0
        mean_psnr = 0
        inp_mean_l1 = 0
        inp_mean_l2 = 0
        inp_mean_psnr = 0
        
        with open(f'{out_dir}/metrics.csv', 'w') as fid:
            fid.write('slice,pixels_per_slice,l1,l2,psnr,inp_l1,inp_l2,inp_psnr\n')
        
        # loop through slices
        for testidx in indices:
            inputs, targets = test_generator.__getitem__(testidx)
            #print(f'Testing image {testidx} of {len(test_generator)}')

            # move input to device
            for idx,inp in enumerate(inputs):
                inputs[idx] = inp.to(device)
            targets = targets.to(device)
            with torch.set_grad_enabled(False):
                pred = self(inputs).detach() # make prediction

            # calculate metrics for this slice
            l1_slc = mae(targets,pred)
            l2_slc = torch.sqrt(mse(targets,pred))
            psnr_slc = psnr(targets.abs(), pred.abs())
            mean_l1 += l1_slc / len(indices)
            mean_l2 += l2_slc / len(indices)
            mean_psnr += psnr_slc / len(indices)

            inp_l1_slc = mae(targets, inputs[0])
            inp_l2_slc = torch.sqrt(mse(targets, inputs[0]))
            inp_psnr_slc = psnr(targets.abs(), inputs[0].abs())
            inp_mean_l1 += inp_l1_slc / len(indices)
            inp_mean_l2 += inp_l2_slc / len(indices)
            inp_mean_psnr += inp_psnr_slc / len(indices)

            # print slice metrics to csv
            with open(f'{out_dir}/metrics.csv', 'a') as fid:
                fid.write(f'{testidx},{torch.numel(targets)},{l1_slc},{l2_slc},{psnr_slc},')
                fid.write(f'{inp_l1_slc},{inp_l2_slc},{inp_psnr_slc}\n')

            print(f'slice {testidx}, {torch.numel(targets)} pixels: l1={l1_slc:.3f}, l2={l2_slc:.3f}, psnr={psnr_slc:.3f}\n')


            
        print(f'num_slices: {len(indices)}')
        print(f'pixels_per_slice: {torch.numel(targets)}')
        print(f'mean_l1: {mean_l1}')
        print(f'mean_l2: {mean_l2}')
        print(f'mean_psnr: {mean_psnr}')

        with open(f'{out_dir}/metrics.csv', 'a') as fid:
            fid.write(f'MEAN,,{mean_l1},{mean_l2},{mean_psnr}, ')
            fid.write(f'{inp_mean_l1},{inp_mean_l2},{inp_mean_psnr}\n')

        
