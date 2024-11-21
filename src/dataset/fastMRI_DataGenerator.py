import numpy as np
import h5py as h5py
import scipy.stats as scs
import unittest
import os

# pytorch
import torch
from torch.utils.data import Dataset

from utils.utils import *
from operators.A_functions import *

# debugging
import pdb
import matplotlib.pyplot as plt

            
########################################## fastMRI ############################################
    
class GeneratorFASTMRI(Dataset):
    'Generates complex-valued data with raw data (for data consistency) for pytorch model'

    def __init__(self, h5_path_list, kspace_dims, h5_slices=128, batch_size=32, shuffle=True, test=False):
        'Initialization'
        self.dtype = torch.complex64
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h5_path_list = h5_path_list
        self.test = test
        self.nlin, self.ncol = kspace_dims

        # get number of h5 files
        self.h5_slices = h5_slices
        self.n_files = len(self.h5_path_list)
        self.n_samples = self.n_files * self.h5_slices # number of slices (128 per h5 file)
        self.n_batches = self.__len__()


        # prepare batch indexing
        self.on_epoch_end()

    def _load_h5_file_with_data(self, file_idx):
        """Method for loading .h5 files:
        """
        h5file = h5py.File(self.h5_path_list[file_idx], 'r')
        nx = h5file['target'].shape[2]

        target = torch.from_numpy(h5file['target'][:])
        smaps = torch.from_numpy(h5file['smaps'][:])
        # noisy = torch.from_numpy(h5file['noisy'][:])
        # kspace = torch.from_numpy(h5file['kspace'][:])
        mask = torch.from_numpy(h5file['mask'][:])

        h5file.close()

        # return [noisy, kspace, mask, smaps], target
        return smaps, target

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

        
    def __get_traj_dcf__(self,ncol=None,nlin=None):
        'Returns an example trajectory, dcf, and dict'
        if ncol is None:
            ncol = self.ncol
        if nlin is None:
            nlin = self.nlin
        traj, dcf = calculate_radial2d_traj_and_dcf(ncol, nlin)
        return torch.from_numpy(traj).to(self.dtype), torch.from_numpy(dcf).to(self.dtype)
                      
    def _normalize(self, targets):
        'Normalize each slice by input max'
            
        for slc in range(targets.shape[0]):
            M = targets[slc,].abs().max()
            targets[slc,] = targets[slc,] / M

        return targets
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        file_idx = self.file_indexes[(index * self.batch_size) // self.h5_slices] # which file to open
        # if this is the first slice of that file, or if in test mode, open it
        if (index*self.batch_size)%self.h5_slices == 0 or self.test:
            # [self.noisy, self.kspace, self.mask, self.smaps], self.target = self._load_h5_file_with_data(file_idx)
            self.smaps, self.target = self._load_h5_file_with_data(file_idx)

        og_idx = np.arange((index*self.batch_size)%self.h5_slices, (index*self.batch_size)%self.h5_slices+self.batch_size)
        sidx = self.slice_indexes[og_idx]

        
        # send to gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [kspace, noisy, targets] = self._normalize(self.kspace[sidx,...], self.noisy[sidx,...], self.target[sidx,...])
        targets = self._normalize(self.target[sidx,...]).to(device)

        
        # image shape
        [_, _, nx, ny] = targets.shape
        # generate trajectory and randomly rotate it
        traj, dcf = self.__get_traj_dcf__(self.ncol, self.nlin)
        # rotation = np.exp(1j*2*np.pi*np.random.rand()) # random rotation optional
        # traj = traj * rotation

        # initialize nufft operators
        A = RadialMulticoilForwardOp([self.nlin, self.ncol], [nx, ny], traj, dcf)
        AH = RadialMulticoilAdjointOp([self.nlin, self.ncol], [nx, ny], traj, dcf)

        
        # generate random undersampling mask
        mask = torch.zeros([self.batch_size, self.nlin, self.ncol]).to(device)
        for slc in range(self.batch_size):
            R = np.random.choice([2,3,4,5,6]) # under-sampling factor
            mask[slc,::R,:] = 1 # uniform under-sampling
            # mask[slc,np.random.choice(self.nlin, int(self.nlin / R), replace=False),:] = 1 # random under-sampling
            
        
        # generate kspace for DC and noisy input
        kspace = A(targets.to(device), torch.ones(mask.shape).to(device), self.smaps[sidx,...].to(device))
        noisy = AH(kspace.to(device), mask, self.smaps[sidx,...].to(device))
        prescan_norm = torch.ones([targets.shape[0], targets.shape[2], targets.shape[3]])
        return [noisy, kspace, mask, self.smaps[sidx,...].to(device)], targets, prescan_norm


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.file_indexes = np.arange(self.n_files)
        self.slice_indexes = np.arange(self.h5_slices)
        if self.shuffle == True:
            np.random.shuffle(self.file_indexes)
            np.random.shuffle(self.slice_indexes)

