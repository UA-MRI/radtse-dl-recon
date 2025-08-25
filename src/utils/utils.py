import torchkbnufft as tkbn
import numpy as np
import torch
import pdb
import os

import pytorch_finufft

import h5py as h5py
import scipy.stats as scs

from utils.utils import *
from utils.espirit import espirit
import matplotlib.pyplot as plt
from time import time

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class KBNuFFT:
    def __init__(self, traj, im_size, dcf=None, input_traj_mode='grid', ncol=None, nlin=None, normalize=True):
        # convert trajectory in the format of the NuFFT API
        # convert to list if given a grid
        ## TODO: allow ktraj to be given as a tensor
        if torch.is_tensor(traj):
            traj = traj.cpu().numpy()
        if input_traj_mode == 'grid':
            ktraj = np.stack([np.imag(traj.transpose().flatten()),
                              np.real(traj.transpose().flatten())])
            ktraj = ktraj * (np.pi / np.max(ktraj))
            # in Matlab the convention is that the trajectory is in [-0.5, 0.5],
            # while here it is in [-pi, pi], so we multiply by pi/max 
            self.ncol, self.nlin = traj.shape
        else: # ncol, nlin input required for list type trajectory
            ktraj = traj
            self.ncol = ncol
            self.nlin = nlin

        # attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ktraj = torch.tensor(ktraj).to(self.device)
        self.dcf = dcf
        if self.dcf is not None:
            if torch.is_tensor(self.dcf):
                self.dcf = torch.transpose(dcf, -1, -2).to(self.device)
            else:
                self.dcf = torch.transpose(torch.tensor(dcf).to(self.device),-1,-2)
        self.im_size = im_size
        self.nufft_ob = tkbn.KbNufft(im_size=im_size, dtype=torch.complex64, device=self.device)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, dtype=torch.complex64, device=self.device)
        self.do_normalize = normalize
        self.norm_val = np.sqrt(np.prod(self.im_size))

    # forward NUFFT
    def forward_op(self, x):
        nslc, ncha, nx, ny = x.shape
        x = x.to(self.device)
        kdata = self.nufft_ob(x, self.ktraj)
        
        if self.do_normalize:
            kdata = kdata / self.norm_val
        
        return torch.reshape(kdata, (nslc, ncha, self.nlin, self.ncol))

    # adjoint NUFFT
    def adjoint_op(self, data):
        nslc, ncha, nlin, ncol = data.shape # get dims
        data = data.to(self.device)
        data = torch.reshape(data,(nslc, ncha, ncol*nlin))
        # density compensation
        if self.dcf is not None:
            if self.dcf.dim() > 2: # if dcf has a slice dimension
                data = data * torch.reshape(self.dcf, [nslc,ncol*nlin]).unsqueeze(1)
            else: # if using the same dcf for all slices and channels
                data = data * torch.reshape(self.dcf, [ncol*nlin])
        img = self.adjnufft_ob(data, self.ktraj) # perform transform
        # normalize
        if self.do_normalize:
            img = img / self.norm_val
        
        return img
    

class FINuFFT:
    def __init__(self, traj, im_size, dcf=None, input_traj_mode='grid', ncol=None, nlin=None, normalize=True):
        # convert trajectory in the format of the NuFFT API
        # convert to list if given a grid
        ## TODO: allow ktraj to be given as a tensor
        if torch.is_tensor(traj):
            traj = traj.cpu().numpy()
        if input_traj_mode == 'grid':
            ktraj = np.stack([np.imag(traj.transpose().flatten()),
                              np.real(traj.transpose().flatten())])
            ktraj = ktraj * (np.pi / np.max(ktraj)) 
            # in Matlab the convention is that the trajectory is in [-0.5, 0.5],
            # while here it is in [-pi, pi], so we multiply by pi/max
            self.ncol, self.nlin = traj.shape[-2:]
        else: # ncol, nlin input required for list type trajectory
            ktraj = traj
            self.ncol = ncol
            self.nlin = nlin

        # attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.complex64
        self.ktraj = torch.tensor(ktraj).to(self.device, torch.float32)
        self.dcf = dcf
        if self.dcf is not None:
            if torch.is_tensor(self.dcf):
                self.dcf = torch.transpose(dcf, -1, -2).to(self.device)
                # self.dcf = dcf.to(self.device)
            else:
                self.dcf = torch.transpose(torch.tensor(dcf).to(self.device),-1,-2)
                # self.dcf = torch.tensor(dcf).to(self.device)
        self.im_size = tuple(im_size)
        ## TODO: FIGURE OUT WHY OUTPUT DOESNT STAY COMPLEX64
        
        self.do_normalize = normalize
        self.norm_val = np.sqrt(np.prod(self.im_size))

    # forward NUFFT
    def forward_op(self, x):
        nslc, ncha, nx, ny = x.shape
        x = x.to(self.device)

        # perform transform
        kdata = pytorch_finufft.functional.finufft_type2(self.ktraj, x, modeord=0, isign=-1)
        
        if self.do_normalize:
            kdata = kdata / self.norm_val
        
        return torch.reshape(kdata, (nslc, ncha, self.nlin, self.ncol))

    # adjoint NUFFT
    def adjoint_op(self, data):
        nslc, ncha, nlin, ncol = data.shape # get dims
        data = data.to(device=self.device, dtype=self.dtype)
        data = torch.reshape(data,(nslc, ncha, ncol*nlin))
        # density compensation
        if self.dcf is not None:
            if self.dcf.dim() > 2: # if dcf has a slice dimension
                data = data * torch.reshape(self.dcf, [nslc,ncol*nlin]).unsqueeze(1)
            else: # if using the same dcf for all slices and channels
                data = data * torch.reshape(self.dcf, [ncol*nlin])

        # perform transform
        img = pytorch_finufft.functional.finufft_type1(self.ktraj, data, self.im_size, modeord=0, isign=1)
        
        # normalize
        if self.do_normalize:
            img = img / self.norm_val
        
        return img
    
def calculate_radial2d_traj_and_dcf(ncol, nlin, xymax=0.5):
    ############################################
    # calculate radial2d and density compensation matrix
    # traj, dcf = calculate_radial2d_traj_and_dcf(ncol, nlin)
    # traj = [ncol, nlin], complex kspace coordinates
    # dcf = [ncol, nlin], radial density compensation
    # Brian Toner, 2022/05/23
    ###########################################
    if nlin % 2:
        angle_range = np.pi * 2
    else:
        angle_range = np.pi

    ## build trajectories and dcf
    traj = np.zeros([ncol, nlin],dtype=complex)
    dcf = np.zeros([ncol, nlin])
    for lin in range(nlin):
        cur_theta = lin * angle_range / nlin

        for col in np.linspace(-ncol*xymax,ncol*xymax-1,ncol):
            # trajectory
            ky = col * np.cos(cur_theta)
            kx = -col * np.sin(cur_theta)
            traj[int(col+ncol/2), lin] = (kx + 1j * ky) / ncol
            # dcf
            if (col == 0):
                dcf[int(col+ncol/2),lin] = angle_range / (4*nlin)
            else:
                dcf[int(col+ncol/2),lin] = angle_range * np.abs(col) / nlin
    return [traj, dcf]

def PC2Contrast(PCs, basis, inverse=False):
    ''' project from PC space to TE space (or inverse) '''
    etl, npc = basis.shape
    
    _, nx, ny = PCs.shape

    if not inverse:
        PC_vec = torch.reshape(PCs, (npc, nx*ny))
        mc_vec = torch.matmul(basis, PC_vec)
        multi_contrast = torch.reshape(mc_vec, (etl, nx, ny))
    else:
        PC_vec = torch.reshape(PCs, (etl, nx*ny))
        mc_vec = torch.matmul(torch.permute(basis, (1,0)), PC_vec)
        multi_contrast = torch.reshape(mc_vec, (npc, nx, ny))

    return multi_contrast


def load_dictionary(dictionary_path, idx=0):
    ''' READ dictionary from an h5 file'''
    h5file = h5py.File(dictionary_path, 'r')
    D = {}
    D['u'] = np.transpose(h5file['u'][idx], (1,0))
    D['s'] = np.transpose(h5file['s'][idx], (1,0))
    D['magnetization'] = np.transpose(h5file['magnetization'][idx], (1,0))
    D['normalization'] = np.transpose(h5file['normalization'][idx], (1,0))
    D['lookup_table'] = np.transpose(h5file['lookup_table'][idx], (1,0))
    D['TE'] = np.squeeze(np.array(h5file['TE'][idx]))
    D['ETL'] = np.array(h5file['ETL'][idx])
    D['T1'] = np.array(h5file['T1'][idx])
    D['fcoherence'] = np.squeeze(np.array(h5file['fcoherence'][idx]))
    D['FAdeg'] = np.array(h5file['FAdeg'][idx])
    D['alpha'] = np.array(h5file['alpha'][idx])

    h5file.close()
    return D

def load_labels(roi_path, idx=0, img_dims=None, anatomy='abdomen'):
    ''' READ ROIs from an h5 file'''
    h5file = h5py.File(roi_path, 'r')
    roi = {}
    if anatomy == 'brain':
        roi['white_matter'] = h5file['WhiteMatter'][...,idx].astype(bool)
        roi['gray_matter'] = h5file['GrayMatter'][...,idx].astype(bool)
        roi['csf'] = h5file['CSF'][...,idx].astype(bool)
        roi['cerebellum'] = h5file['Cerebellum'][...,idx].astype(bool)
        if img_dims is not None:
            roi['white_matter'] = crop_img(roi['white_matter'], img_dims)
            roi['gray_matter'] = crop_img(roi['gray_matter'], img_dims)
            roi['csf'] = crop_img(roi['csf'], img_dims)
            roi['cerebellum'] = crop_img(roi['cerebellum'], img_dims)
    elif anatomy == 'leg':
        roi['muscle'] = h5file['Muscle'][...,idx].astype(bool)
        if img_dims is not None:
            roi['muscle'] = crop_img(roi['muscle'], img_dims)
    else:
        roi['liver'] = h5file['Liver'][...,idx].astype(bool)
        roi['Rkidney'] = h5file['RKidney'][...,idx].astype(bool)
        roi['Lkidney'] = h5file['LKidney'][...,idx].astype(bool)
        roi['spleen'] = h5file['Spleen'][...,idx].astype(bool)
        roi['muscle'] = h5file['Muscle'][...,idx].astype(bool)

        if 'Body' in h5file.keys():
            roi['body'] = h5file['Body'][...,idx].astype(bool)

        if img_dims is not None:
            roi['liver'] = crop_img(roi['liver'], img_dims)
            roi['Rkidney'] = crop_img(roi['Rkidney'], img_dims)
            roi['Lkidney'] = crop_img(roi['Lkidney'], img_dims)
            roi['spleen'] = crop_img(roi['spleen'], img_dims)
            roi['muscle'] = crop_img(roi['muscle'], img_dims)
            
            if 'body' in roi.keys():
                roi['body'] = crop_img(roi['body'], img_dims)
    
    h5file.close()
    return roi


def gen_t2_maps(alpha, D, x_range=[0,-1], y_range=[0,-1]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = alpha.to(device)
    alpha = alpha[:,x_range[0]:x_range[1],y_range[0]:y_range[1]]
    [npc, ny, nx] = alpha.shape
    alpha = torch.reshape(alpha, [npc, ny*nx])
    alpha = alpha / (torch.norm(alpha, p=2, dim=0) + 1e-8)

    # 1D method
    Dmag = torch.permute(torch.from_numpy(D['magnetization']),[1,0]).to(device,torch.complex64)
    T2_table = torch.from_numpy(D['lookup_table'][:,1]).to(device)

    ip = torch.inner(Dmag, alpha.H)

    [fit_map, idx] = torch.max(ip.abs(), dim=0)
    T2map = torch.take(T2_table, idx)

    T2map = torch.reshape(T2map, [ny, nx])
    fit_map = torch.reshape(fit_map, [ny, nx])
    
    return T2map.cpu().numpy(), fit_map


# N-dimensional Cartesian FFT (considering ifftshift/fftshift operations)
def fftnc(x, axes=(0, 1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

# N-dimensional IFFT (considering ifftshift/fftshift operations)
def ifftnc(x, axes=(0, 1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x


# N-dimensional Cartesian FFT (considering ifftshift/fftshift operations)
def fftnc_torch(x, axes=(0, 1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=ax), dim=ax), dim=ax)
    return x

# N-dimensional IFFT (considering ifftshift/fftshift operations)
def ifftnc_torch(x, axes=(0, 1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=ax), dim=ax), dim=ax)
    return x

def circle_mask(d):
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    r = d // 2
    mat = np.zeros((d, d))
    rx , ry = d/2, d/2
    for row in range(d):
        for col in range(d):
            dist = distance(rx, rx, row, col)
            if dist < r:
                mat[row, col] = 1
    return mat


def crop_img(img, img_dims):
    # crop img to image size
    if img_dims != img.shape[-2:-1]:
        center_x = int(img.shape[-2] // 2)
        center_y = int(img.shape[-1] // 2)
        xmin = int(center_x - img_dims[0] // 2)
        xmax = int(center_x + img_dims[0] // 2)
        ymin = int(center_y - img_dims[1] // 2)
        ymax = int(center_y + img_dims[1] // 2)
        img = img[xmin:xmax, ymin:ymax]
    return img

# copied from different version of torch.optim.swa_utils.py
def get_ema_multi_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param_list, current_param_list, _):
        for p_ema, p_model in zip(ema_param_list, current_param_list):
            p_ema.copy_(p_ema * decay + p_model * (1 - decay))
    
    def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
        return (1-decay) * averaged_model_parameter + decay * model_parameter
    

    return ema_avg

def colorLogRemap(oriCmap, loLev, upLev):
    # colorLogRemap: lookup of the original color map table according to a "log-like" curve.
    #   The log-like curve contains a linear part and a logarithmic part; the size of the parts
    #   depends on the range (loLev,upLev) 

    #   Arguments:
    #       oriCmap     original colormap, provided as a N*3 matrix
    #       loLev       lower level of the range to be displayed
    #       upLev       upper level of the range to be displayed
    #   Returns:  modified colormap
    assert upLev > 0 # upper level must be positive
    assert upLev > loLev # upper level must be larger than lower level
    
    mapLength = oriCmap.shape[0]
    eInv = np.exp(-1.0)
    aVal = eInv * upLev
    mVal = np.max([aVal, loLev])
    bVal = (1.0 / mapLength) + (aVal >= loLev) * ((aVal - loLev) / (2 * aVal - loLev))
    bVal = bVal+0.0000001   # This is to ensure that after some math, we get a figure that rounds to 1 ("darkest valid color")
                            # rather than to 0 (invalid color). Note that bVal has no units, so 1E-7 is always a small number    
    logCmap = np.zeros_like(oriCmap)
    logCmap[0, :] = oriCmap[0, :]
    logPortion = 1.0 / (np.log(mVal) - np.log(upLev))

    for g in range(1, mapLength):
        f = 0.0
        x = (g+1) * (upLev - loLev) / mapLength + loLev
        if x > mVal:
            # logarithmic segment of the curve
            f = mapLength * ((np.log(mVal) - np.log(x)) * logPortion * (1 - bVal) + bVal)
        else:
            if (loLev < aVal) and (x > loLev):
                # linear segment of the curve
                f = mapLength * ((x - loLev) / (aVal - loLev) * (bVal - (1.0 / mapLength))) + 1.0
            if (x <= loLev) :
                # lowest valid color
                f = 1.0
        # lookup from original color map
        logCmap[g, :] = oriCmap[np.min([mapLength-1, np.floor(f).astype(int)]), :]
    # Return modified colormap
    return logCmap

    
def relaxationColorMap(maptype, x, loLev, upLev):
    # [xClip, lutCmap] = relaxationColorMap(maptype, x, loLev, upLev)

    # RelaxationColorMap: acts in two ways:
    #   1. generate a colormap to be used on display, given image type 
    #      (which must be one of 
    #       "T1","R1","T2","T2*","R2","R2*","T1rho","T1ρ","R1rho","R1ρ","t1","r1","t2","t2*","r2","r2*","t1rho","t1ρ","r1rho","r1ρ")
    #      and given the range of the image to be displayed;
    #   2. generates a 'clipped' image, which is a copy of the input image except that values are clipped to the lower level,
    #      while respecting the special value of 0 (which has to map to the "invalid" color)
    # INPUTS:
    #    maptype: a string from aformentioned series, e.g. "T1"  or "R2"
    #    x      : ND array containing the image to be displayed
    #    loLev  : lower level of the range to be displayed
    #    upLev  : upper level of the range to be displayed
    # OUTPUTS:  
    #    xClip  : value-clipped image with the same size as x
    #    lutCmap: 256 by 3 colormap to be used in image-display functions (in Colors.RGB format)

    # Original version by M. Fuderer, UMC Utrecht; using ChatGPT on RelaxationColor.jl
    # 3-4-2024, D.Poot, Erasmus MC: bugfixes and substantial performance improvement.
    
    # ADAPTED FROM MATLAB VERSION BY BRIAN TONER
    
    if maptype in ['T1','R1']:
        colortable = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/lipari.npy')
    elif maptype in ['T2', 'T2*', 'R2', 'R2*', 'T1rho', 'T1ρ', 'R1rho', 'R1ρ']:
        colortable = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/navia.npy')
    else:
        print('ERROR: Expect "T1", "T2", "R1", or "R2" as maptype')
        exit()

    if maptype[0] == 'R':
        colortable = colortable[::-1,:] # flip for R1 or R2 map

    
    colortable[0,:-1] = 0 # set 'invalid value' color
    
    # modification of the image to be displayed
    eps = (upLev - loLev) / colortable.shape[0]

    # this if statement never made sense to me, so I got rid of the 2nd option
    if loLev < 0:
        xClip = (x < eps) * (loLev - eps) + (x >= eps) * x
    else:
        xClip = (x <  eps) * (loLev - eps) + (x >= eps) * ( (x < loLev + eps) * (loLev + 1.5 *eps ) +  (x >= loLev + eps) * x)

    lutCmap = colorLogRemap(colortable, loLev, upLev)
    
    return [xClip, ListedColormap(lutCmap)]


def compute_csm(img, kSize=4, CalibSize=16):
    # Compute CSMs with ESPIRIT
    # kSize is kernel size, CalibSize is the calibration region size

    # project to cartesian kspace
    kspaceCX = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img.cpu().numpy(), axes=(-2, -1))), axes=(-2, -1))

    eigThresh_k = 0.02 # 0.02  # threshold of eigenvectors in k-space
    eigThresh_im = 2e-16 #0  # 0.95 # threshold of eigenvectors in image space

    nmaps = 1 # number of maps to compute (keep higher eigenvalues)
    # if set >0 then there is a threshold under which CSM values are just zeros.
    # Not sure if in practice it is better to have eigThresh_im=0 or eigThresh_im=095 e.g.
    esp = espirit(kspaceCX.transpose(2, 3, 0, 1), kSize, CalibSize, eigThresh_k, eigThresh_im, m=nmaps)
    cmap = esp[:, :, :, :, 0]  # we keep only the first set of CSMs (highest eigenvalue)
    cmap = cmap.transpose(2, 3, 0, 1)  # nufft_ob assume the batch and coil dimensions are the first two dimensions
            
    return cmap
