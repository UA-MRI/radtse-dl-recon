import numpy as np
import torch
import pdb

# Merlin 
import merlinth.layers.mri
from merlinth.layers.fft import fft2, fft2c, ifft2, ifft2c

# my stuff
from utils.utils import *

# debug
import pdb
from time import time

class MulticoilForwardOp(torch.nn.Module):
    ''' Multicoil cartesian '''
    def __init__(self, center=False, coil_axis=1, channel_dim_defined=True):
        super().__init__()
        # forward fft
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, image, mask):
        # apply forward fft and mask output
        if self.channel_dim_defined:
            kspace = self.fft2(image[:,0,...])
        else:
            kspace = self.fft2(image[:,0,...])
        masked_kspace = kspace * mask
        return masked_kspace

class MulticoilAdjointOp(torch.nn.Module):
    ''' Multicoil cartesian '''
    def __init__(self, center=False, coil_axis=1, channel_dim_defined=True):
        super().__init__()
        # inverse fft
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, kspace, mask):
        # mask kspace and apply ifft
        masked_kspace = kspace * mask
        img = self.ifft2(masked_kspace)
        if self.channel_dim_defined:
            return torch.unsqueeze(img, self.coil_axis)
        else:
            return img
        

class RadialForwardOp(torch.nn.Module):
    ''' radial single coil '''
    def __init__(self, kspace_dims, img_dims, package='finufft'):
        super().__init__()
        # initialize radial nufft
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.traj, self.dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        if package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, image, mask):
        # apply forward nufft, mask output
        kspace = self.nufft.forward_op(image).to(torch.complex64)
        masked_kspace = kspace * mask
        return masked_kspace

class RadialAdjointOp(torch.nn.Module):
    ''' radial single coil '''
    def __init__(self, kspace_dims, img_dims, package='finufft'):
        super().__init__()
        # initialize adjoint radial nufft
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.traj, self.dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        if package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, kspace, mask):
        # mask kspace and apply adjoint radial nufft
        masked_kspace = kspace * mask
        img = self.nufft.adjoint_op(masked_kspace).to(torch.complex64)

        
        return img
    
class RadialMulticoilForwardOp(RadialForwardOp):
    ''' multicoil radial -- use this for composite images '''
    def __init__(self, kspace_dims, img_dims, traj=None, dcf=None, package='finufft'):
        super().__init__(kspace_dims, img_dims, package=package)
        # only initialize dimensions
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.package = package
        
        # calculate or set traj and dcf
        if traj is None and dcf is None:
            traj, dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        elif traj is None:
            traj, _ = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        elif dcf is None:
            _, dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
            
        self.traj = traj.squeeze()
        self.dcf = dcf.squeeze()

        # initialize operator
        if self.package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif self.package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, image, mask, smaps, D=None):
        # apply forward nufft
        coil_imgs = image * smaps # coil uncombine
        kspace = self.nufft.forward_op(coil_imgs).to(torch.complex64) # nufft
        # mask output
        while mask.ndim < kspace.ndim:
            mask = mask.unsqueeze(1)
        kspace = kspace * mask
        
        return kspace

class RadialMulticoilAdjointOp(RadialAdjointOp):
    ''' multicoil radial -- use this for composite images '''
    def __init__(self, kspace_dims, img_dims, traj=None, dcf=None, package='finufft'):
        super().__init__(kspace_dims, img_dims, package=package)
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.package = package
        
        # calculate or set traj and dcf
        if traj is None and dcf is None:
            traj, dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        elif traj is None:
            traj, _ = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
        elif dcf is None:
            _, dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)

        self.traj = traj.squeeze()
        self.dcf = dcf.squeeze()

        # initialize operator
        if self.package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif self.package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, kspace, mask, smaps, D=None):
        device = kspace.device # 'cuda' or 'cpu'
        
        # mask kspace 
        while mask.ndim < kspace.ndim:
            mask = mask.unsqueeze(1)
        kspace = kspace * mask
        
        # apply adjoint nufft
        coil_imgs = self.nufft.adjoint_op(kspace).to(torch.complex64).to(device) # anufft
        img = torch.sum(coil_imgs * torch.conj(smaps), axis = 1) # coil combine
        
        return img.unsqueeze(1)

    
class RADTSE_ForwardOp(RadialMulticoilForwardOp):
    ''' multicoil multi-PC radial -- for PC images '''
    def __init__(self, kspace_dims, img_dims, traj, dcf, package='finufft'):
        super().__init__(kspace_dims, img_dims, package=package)
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.package = package
        
        self.traj = traj.squeeze()
        self.dcf = dcf.squeeze()

        # initialize operator
        if self.package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif self.package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, image, mask, smaps, D):
        device = image.device # 'cuda' or 'cpu'
        D = D.to(device) # move to device
        nslc, npc, nx, ny = image.shape # get image dims
        ncha = smaps.shape[1] # get ncoils
        etl = D.shape[-2] # get etl

        # project PC images to TE images
        img = torch.empty((nslc, etl, nx, ny), dtype=torch.complex64, device=device)
        for slc in range(nslc):
            img[slc,] = PC2Contrast(image[slc,], D[slc,])
        img = torch.permute(img, [0,2,3,1])

        # initialize kspace tensor
        kspace = torch.empty((nslc, ncha, etl, self.nlin // etl, self.ncol),
                             dtype=torch.complex64, device=device)

        # NEW method parallelizes across batch AND TE
        # this is a bit faster if memory allows, but more memory hungry
        if self.nlin <= 384:
            # do forward NUFFT for each TE image
            coil_imgs = img.unsqueeze(1) * smaps.unsqueeze(-1)
            coil_imgs = torch.reshape(torch.permute(coil_imgs, [4,0,1,2,3]), [-1,ncha,nx,ny])
            kspace_ee = self.nufft.forward_op(coil_imgs).to(torch.complex64).to(device) # nufft
            kspace_ee = torch.reshape(kspace_ee, [etl, nslc, ncha, etl, -1, self.ncol])
            for ee in range(etl):
                kspace[:,:,ee,:,:] = kspace_ee[ee,:,:,ee,:,:]
                
            kspace = torch.reshape(kspace, [nslc, ncha, self.nlin, self.ncol])
        
        ## OLD METHOD does batch in parallel but TE in series
        # a bit slower but requires less memory -- use it when nlin is high
        else:
            # do forward NUFFT for each TE image
            for ee in range(etl):
                coil_imgs = img[...,ee].unsqueeze(1) * smaps # coil uncombine
                kspace_ee = self.nufft.forward_op(coil_imgs).to(torch.complex64).to(device) # nufft
                kspace[:,:,ee,:,:] = torch.reshape(kspace_ee, [nslc, ncha, etl, -1, self.ncol])[:,:,ee,:,:]

            kspace = torch.reshape(kspace, [nslc, ncha, self.nlin, self.ncol])
        
        # mask output
        while mask.ndim < kspace.ndim:
            mask = mask.unsqueeze(1)
        kspace = kspace * mask
        
        return kspace 

class RADTSE_AdjointOp(RadialMulticoilAdjointOp):
    ''' multicoil multi-PC radial -- for PC images '''
    def __init__(self, kspace_dims, img_dims, traj, dcf, package='finufft'):
        super().__init__(kspace_dims, img_dims, package=package)
        self.nlin, self.ncol = kspace_dims
        self.img_dims = img_dims
        self.package = package
        
        self.traj = traj.squeeze()
        self.dcf = dcf.squeeze()

        # initialize operator
        if self.package == 'kbnufft':
            self.nufft = KBNuFFT(self.traj, self.img_dims, self.dcf)
        elif self.package == 'finufft':
            self.nufft = FINuFFT(self.traj, self.img_dims, self.dcf)

    def forward(self, kspace, mask, smaps, D):
        device = kspace.device # 'cuda' or 'cpu'
        D = D.to(device) # move to device
        nslc, ncha, nlin, ncol = kspace.shape # get kspace dims
        etl, npc = D.shape[-2:] # get etl and npc
            
        # make sure mask and kspace are same dim
        while mask.ndim < kspace.ndim:
            mask = mask.unsqueeze(1)
    
        # calculate TE images from kspace
        ## NEW METHOD PARALLELIZES ACROSS TE
        # a bit faster but more memory hungry
        if nlin <= 384:
            ee_mask_full = torch.zeros_like(mask).unsqueeze(0).repeat([etl,1,1,1,1])
            for ee in range(etl):
                ee_mask = torch.reshape(torch.zeros_like(mask), [nslc, 1, etl, -1, ncol])
                ee_mask[:,:,ee,:,:] = 1
                ee_mask = torch.reshape(ee_mask, mask.shape) * mask
                ee_mask_full[ee,] = ee_mask
            masked_kspace = torch.reshape(ee_mask_full * kspace.unsqueeze(0), [-1, ncha, nlin, ncol])
            coil_imgs = self.nufft.adjoint_op(masked_kspace).to(torch.complex64).to(device) # anufft
            coil_imgs = torch.permute(torch.reshape(coil_imgs, [etl, nslc, ncha, *self.img_dims]), [1,2,3,4,0])
            TEs = torch.sum(coil_imgs * torch.conj(smaps).unsqueeze(-1), axis=1) # coil combine
        
        ## OLD METHOD does batch in parallel but TE in series
        # a bit slower but less memory hungry
        else:
            # initialize output
            TEs = torch.empty(nslc, *self.img_dims, etl, dtype=torch.complex64, device=device)
            for ee in range(etl):
                # do transform on masked k-space
                ee_mask = torch.reshape(torch.zeros_like(mask), [nslc, 1, etl, -1, ncol])
                ee_mask[:,:,ee,:,:] = 1
                ee_mask = torch.reshape(ee_mask, mask.shape) * mask
                coil_imgs = self.nufft.adjoint_op(ee_mask * kspace).to(torch.complex64).to(device) # anufft

                TEs[:,:,:,ee] = torch.sum(coil_imgs * torch.conj(smaps), axis=1) # coil combine
                
        # project from TE to PC images
        alpha = torch.empty(nslc, self.img_dims[0], self.img_dims[1], npc, dtype=torch.complex64, device=device)
        for slc in range(nslc):
            alpha[slc,] = TEs[slc,] @ D[slc,]

        return torch.permute(alpha, (0,3,1,2))
