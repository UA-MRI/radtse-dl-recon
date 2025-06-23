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
from time import time, sleep

            
########################################## RADTSE ############################################
    
class DataGeneratorRADTSE(Dataset):
    'Generates complex-valued data with raw data (for data consistency) for pytorch model'

    def __init__(self, h5_path, image_type, phase, config, R=0.5, dcf_method=None, ref_echo=0, prescan_norm_power=1):
        'Initialization'
        
        self.image_type = image_type # composite or pcs
        self.dtype = torch.complex64
        if phase == 'test':
            self.batch_size = 1
        else:
            self.batch_size = config['batch_size']
        self.h5_path = h5_path
        self.normalize = config['normalize'] # normalize the input images
        self.phase = phase # train valid or test
        self.config = config # config parameters
        self.shuffle = phase=='train'
        self.R = R # acceleration for testing
        self.dcf_method = dcf_method # None for ramp dcf or pipe for pipe/menon (1999) implementation
        self.ref_echo = ref_echo # set to different integers <= ETL to create new under-sampling patterns
        self.prescan_norm_power = prescan_norm_power # raise the prescan norm map to this exponent
        
        if config['img_dims'] is None:
            with h5py.File(self.h5_path) as h5file:
                ncol = h5file['kspace'].shape[3]
            self.img_dims = (ncol, ncol)
        else:
            self.img_dims = config['img_dims']
            
        # load data
        self._load_h5_file_with_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        traj, dcf, _ = self.__get_traj_dcf_dict__()

        # if ramp is not being used for input, still use it for target
        if self.dcf_method == 'ramp':
            self.dcf = dcf
        
        # set the adjoint model
        if self.image_type == 'composite':
            # if no dcf used, use ramp for target
            self.AH = RadialMulticoilAdjointOp(kspace_dims=[self.nlin,self.ncol], img_dims=self.img_dims, traj=traj, dcf=dcf) 
            self.target_AH = RadialMulticoilAdjointOp(kspace_dims=[self.nlin,self.ncol], img_dims=self.img_dims, traj=self.traj, dcf=self.dcf)
        else:
            # if no dcf used, use ramp for target
            self.AH = RADTSE_AdjointOp(kspace_dims=[self.nlin,self.ncol], img_dims=self.img_dims, traj=traj, dcf=dcf)
            self.target_AH = RADTSE_AdjointOp(kspace_dims=[self.nlin,self.ncol], img_dims=self.img_dims, traj=self.traj, dcf=self.dcf)
            
        # prepare batch indexing
        self.on_epoch_end()

    def _load_h5_file_with_data(self):
        """Method for loading .h5 files:
        """
        # start timer
        tstart = time()
        self.h5_path_np = self.h5_path.replace('.h5','_np.h5')
        # saving h5 in matlab forces us to split real and imag components, and reverses the dimensions
        # convert matlab to numpy if havent done so yet
        if not os.path.exists(self.h5_path_np): 
            print('Loading from matlab version of h5')
            h5file = h5py.File(self.h5_path, 'r')
            
            # save numpy version for future use
            h5out = h5py.File(self.h5_path_np, 'w')
            
            traj = (h5file['traj'][...,0,0] + 1j*h5file['traj'][...,0,1]).squeeze()
            dcf = h5file['dcf'][...,0]
            D = np.transpose(h5file['dict'][:],(2,1,0))
            print(f'traj, dcf, dict loaded. {time()-tstart:.2f}s elapsed')

            # create datasets
            h5out.create_dataset('traj', data=np.transpose(traj,(0,2,1)))
            h5out.create_dataset('dcf', data=np.transpose(dcf,(0,2,1)))
            h5out.create_dataset('dict', data=D)

            npc = D.shape[-1]
            [ncha, etl, ntr, ncol, nslc,_] = h5file['kspace'].shape
            [_, nx, ny,_,_] = h5file['smaps'].shape
            
            kspace = h5out.create_dataset('kspace', (nslc,ncha,etl,ntr,ncol), np.complex64)
            llr = h5out.create_dataset('llr', (nslc,npc,nx,ny), np.complex64)
            nlr3d = h5out.create_dataset('nlr3d', (nslc,npc,nx,ny), np.complex64)
            smaps = h5out.create_dataset('smaps', (nslc,ncha,nx,ny), np.complex64)
            prescan_norm = h5out.create_dataset('prescan_norm', (nslc,nx,ny), np.float32)

            # copy data over. break into chunks of 1000 slices to not go over memory limits
            for idx in range((nslc // 1000) + 1):
                imin = idx*1000
                imax = np.min([nslc, imin+1000])
    
                print(f'saving slices {imin}-{imax} of {nslc} total')
                
                print(f'saving prescan norm. {time()-tstart:.2f}s elapsed')
                tmp = h5file['prescan_norm'][...,imin:imax]
                prescan_norm[imin:imax,...] = np.transpose(tmp, (2,0,1))
                
                print(f'saving kspace slices. {time()-tstart:.2f}s elapsed')
                tmp = (h5file['kspace'][...,imin:imax,0] + 1j*h5file['kspace'][...,imin:imax,1]).squeeze()
                kspace[imin:imax,...] = np.transpose(tmp, (4,0,1,2,3))
                
                print(f'saving LLR slices. {time()-tstart:.2f}s elapsed')
                tmp = (h5file['llr'][...,imin:imax,0] + 1j*h5file['llr'][...,imin:imax,1]).squeeze()
                llr[imin:imax,...] = np.transpose(tmp, (3,0,1,2))
                
                print(f'saving NLR3D slices. {time()-tstart:.2f}s elapsed')
                tmp = (h5file['nlr3d'][...,imin:imax,0] + 1j*h5file['nlr3d'][...,imin:imax,1]).squeeze()
                nlr3d[imin:imax,...] = np.transpose(tmp, (3,0,1,2))
                
                print(f'saving smaps slices. {time()-tstart:.2f}s elapsed')
                tmp = (h5file['smaps'][...,imin:imax,0] + 1j*h5file['smaps'][...,imin:imax,1]).squeeze()
                smaps[imin:imax,...] = np.transpose(tmp, (3,0,1,2))
                
            h5out.close()
            h5file.close()
            print(f'Saved numpy version of h5. {time()-tstart:.2f}s elapsed')

        
        # load numpy version now
        print('Loading from numpy version of h5')
        h5file = h5py.File(self.h5_path_np, 'r')

        # find dims and number os samples/batches
        self.n_samples, self.ncha, self.etl, self.nshot, self.ncol = h5file['kspace'].shape
        self.nlin = self.etl * self.nshot
        _, _, self.nx, self.ny = h5file['smaps'].shape
        self.n_batches = self.__len__()
        self.npc = h5file['dict'].shape[-1]

        # load traj, dcf, dictionary
        self.traj = torch.from_numpy(h5file['traj'][:])
        self.dcf = torch.from_numpy(h5file['dcf'][:])
        self.D = torch.from_numpy(h5file['dict'][:])
        # reshape traj, dcf
        self.traj = torch.reshape(torch.permute(self.traj, [1,0,2]), [self.ncol,-1])
        self.dcf = torch.reshape(torch.permute(self.dcf, [1,0,2]), [self.ncol,-1])
        self.dcf = self.dcf / self.dcf.max() # make max = 1
            
        
        h5file.close()
        
        
    def _normalize(self, noisy, kspace=None):
        'Normalize each slice by input max'
        for slc in range(noisy.shape[0]):
            M = noisy[slc,].abs().max() # max in image space
            noisy[slc,] = noisy[slc,] / M # normalize image
            if kspace is not None: # option to normalize k-space as well
                kspace[slc,] = kspace[slc,] / M

        if kspace is None:
            return noisy
        else:
            return noisy, kspace

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __get_traj_dcf_dict__(self,idx=0):
        'Returns an example trajectory, dcf, and dict'
        # use method from pipe/menon 1999
        if self.dcf_method == 'pipe':
            if self.phase == 'train':
                masks = torch.ones_like(self.get_masks())
            else:               
                masks = self.get_masks()
            masks = torch.permute(masks[0,], [1,0]).cpu()
            
            dcf = compute_pipe_dcf(self.traj * masks, self.ncol, self.etl, self.nshot, self.img_dims, self.image_type)
        # or don't use dcf (all ones)
        elif self.dcf_method == 'ones':
            dcf = torch.ones_like(self.dcf)
        else:
            dcf = self.dcf
        
        return self.traj.to(self.device), dcf.to(self.device), self.D[idx,:,:].to(self.device)

    def __get_cs__(self, index, algorithm='LLR'):
        'Returns CS PC maps'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        cs = torch.zeros((self.batch_size, self.npc, self.nx, self.ny), dtype=self.dtype) # cs reference

        h5file = h5py.File(self.h5_path_np, 'r')
        for i, ID in enumerate(indexes):
            if algorithm.upper() =='LLR':
                cs[i,] = torch.from_numpy(h5file['llr'][ID,])
            elif algorithm.upper() == 'NLR3D':
                cs[i,] = torch.from_numpy(h5file['nlr3d'][ID,])
                
        h5file.close()
        return self._normalize(self.crop_smaps(cs))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # pre-allocate outputs
        kspace = torch.empty((self.batch_size, self.ncha, self.nlin, self.ncol), dtype=self.dtype) # rawdata/k-space for data consistency
        smaps = torch.empty((self.batch_size, self.ncha, self.ncol, self.ncol), dtype=self.dtype) # sensitivity maps
        prescan_norm = torch.ones((self.batch_size, self.nx, self.ny), dtype=self.dtype) # prescan normalization
        D = torch.empty(self.batch_size, self.etl, self.npc, dtype=self.dtype) # signal SVD basis

        h5file = h5py.File(self.h5_path_np, 'r')
        
        # load data
        for i, ID in enumerate(indexes):
            # Get samples
            smaps[i,] = torch.from_numpy(h5file['smaps'][ID,])
            kspace[i,] = torch.reshape(torch.from_numpy(h5file['kspace'][ID,]), [-1, self.ncha, self.nlin, self.ncol])
            D[i,] = torch.from_numpy(h5file['dict'][ID,])
            # raise prescan normalization map to an exponent to adjust its effect on the image (default 1)
            if self.prescan_norm_power:
                prescan_norm[i,] = torch.from_numpy(h5file['prescan_norm'][ID,]).to(self.dtype) ** self.prescan_norm_power

        # crop smaps and prescan norm to image size
        smaps = self.crop_smaps(smaps)
        prescan_norm = self.crop_smaps(prescan_norm.unsqueeze(1))[:,0,]

        # move to device
        kspace = kspace.to(self.device)
        smaps = smaps.to(self.device)
        
        # determine mask
        if self.phase == 'test' and self.R == 1:
            # dont mask data if acceleration is set to 1
            masks = torch.ones([self.batch_size, self.nlin, self.ncol]).to(self.device)
        else:
            # mask data according to self.R
            masks = self.get_masks().to(self.device)

        # get noisy input
        noisy = self.AH(kspace, masks, smaps, D).to(self.device)

        # get target
        if self.phase == 'test':
            cs = self.__get_cs__(index).to(self.device)
            # if in test mode, look for CS recon as target
            if cs.abs().sum() == 0 or torch.isnan(cs).sum() or self.image_type == 'composite' or self.nlin > 384:
                if self.R == 1:
                    # if R = 1 and no CS, then we dont have a target
                    print('No targets found')
                    targets = None
                else:
                    # if R < 1 and no CS, use the R = 1 version as the target
                    print('using R=1 NUFFT as target')
                    targets = self.target_AH(kspace, torch.ones_like(masks), smaps, D).to(self.device)
            else:
                print('using CS recon as target')
                targets = self.crop_smaps(cs)
        else:
            # if in valid or train mode, just use the R = 1 version as the target
            targets = self.target_AH(kspace, torch.ones_like(masks), smaps, D).to(self.device)
        
        # normalize
        if self.normalize:
            noisy, kspace = self._normalize(noisy, kspace) # normalize data
            if targets is not None:
                targets = self._normalize(targets) # normalize targets
        
        h5file.close()
        return [noisy, kspace, masks, smaps, D], targets, prescan_norm

    
    def get_masks(self):
        # generate under-sampling masks
        masks = torch.zeros([self.batch_size, self.etl, self.nshot, self.ncol]).to(self.device)
        lin_mask = np.zeros(self.nshot)
        if self.phase == 'train':
            N = int(np.round(np.random.uniform(2, 2/3*self.nshot))) # randomly select how many TRs
            lin_mask[np.random.choice(self.nshot, N, replace=0)] = 1 # randomly select TRs
        else:
            N = int(np.round(self.R*self.nshot)) # select number of TRs using R
            #### evenly distributed trajectory
            traj = torch.reshape(self.traj, [self.ncol, self.etl, self.nshot])
            angles = np.angle(traj[0,self.ref_echo,:]) # all angles available in trajectory
            angles = angles + np.pi * (angles < 0) # rescale to [0, 2*pi]
            # angle0 = angles[np.random.choice(self.nshot)] # random angle
            angle0 = angles[0] # first angle
            linindx = [] # line index list
            for angidx in range(0,N):
                ang = (angle0 + angidx * np.pi / N) % np.pi  # ideal angle
                linindx += [np.argmin(np.abs(angles-ang))] # closest angle to ang
            
            lin_mask[linindx] = 1 # keep the angles selected



        lin_mask = lin_mask.astype(bool) # convert to bool for indexing
        masks[:,:,lin_mask,:] = 1 # mask for input to network
        
        return torch.reshape(masks, [self.batch_size, self.nlin, self.ncol])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def crop_smaps(self, smaps):
        # crop smaps to image size (we can use this on other images too)
        if self.img_dims != smaps.shape[-2:-1]:
            center_x = int(smaps.shape[-2] // 2)
            center_y = int(smaps.shape[-1] // 2)
            xmin = int(center_x - self.img_dims[0] // 2)
            xmax = int(center_x + self.img_dims[0] // 2)
            ymin = int(center_y - self.img_dims[1] // 2)
            ymax = int(center_y + self.img_dims[1] // 2)
            smaps = smaps[:,:,xmin:xmax, ymin:ymax]
        return smaps

            
    def qc_data_generator(self, A, AH, idx=0, qcdir='/clusterscratch/tonerbp/dlrecon_radtse/qc/'):
        ''' QC check for the data loader '''
        # get inputs, targets
        inputs, targets, _ = self.__getitem__(idx)
        traj, dcf, D = self.__get_traj_dcf_dict__()
        # move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for tmpidx,inp in enumerate(inputs):
            inputs[tmpidx] = inp.to(device)
        # kspace and image space
        k = inputs[1] 
        x = inputs[0]

        # time the adjoint and foward operators
        tstart = time()
        AHk = AH(inputs[1], inputs[2], inputs[3], inputs[4])
        Ax = A(inputs[0], inputs[2], inputs[3], inputs[4])
        print(f'{time()-tstart:.2f}s elapsed')
        
        etl, npc = D.shape[-2:] # get dims
        # for first time, test that our operators are true adjoints
        if idx == 0:
            # need to divide by dcf because out operator automatically multiplies by it
            # its only a true adjoint without the dcf
            AH1 = AH(torch.ones_like(k) / torch.permute(dcf, [1,0]), inputs[2], inputs[3], inputs[4])

            # if A and AH are adjoints, these will be the same
            print(torch.inner(torch.conj(torch.reshape(AH1, [-1,1]).squeeze()),
                              torch.reshape(x, [-1,1]).squeeze()))
            print(torch.inner(torch.conj(torch.reshape(torch.ones_like(k), [-1,1]).squeeze()),
                              torch.reshape(Ax, [-1,1]).squeeze()))

        # check that our new AHk matches the AHk the data loader gave
        loss = torch.norm(inputs[0] - AHk).item()
        # save qc images
        if True: # loss > 1 or idx == 0:
            AHAx = AH(Ax, *inputs[2:])
            AAHk = A(AHk, *inputs[2:])
            Ax = A(inputs[0], *inputs[2:]) 

            cha = 0
            for ii in [0,1,3]:
                if ii == 1 or ii == 2:
                    inp = np.transpose(inputs[ii][0,cha,].reshape(self.nlin,self.ncol).abs().cpu().numpy(),(1,0))
                else:
                    inp = np.transpose(inputs[ii][0,cha,].abs().cpu().numpy(),(1,0))
                plt.imsave(f'{qcdir}inputs_{ii}.png',inp,cmap='gray',vmax=np.percentile(inp,99))

            mk = np.transpose((inputs[1]*inputs[2])[0,cha,].reshape(self.nlin,self.ncol).abs().cpu().numpy(),(1,0))
            plt.imsave(f'{qcdir}masked_k.tiff', mk, vmin=0, vmax=np.percentile(mk,99),cmap='gray')
            if targets is not None:
                tar = np.transpose(targets[0,cha,].abs().cpu().numpy(),(1,0))
                plt.imsave(f'{qcdir}targets.png', tar,cmap='gray',vmax=np.percentile(tar,99))
    
            Ax = np.transpose(Ax[0,cha,].reshape(self.nlin,self.ncol).abs().cpu().numpy(),(1,0))
            plt.imsave(f'{qcdir}Ax.png', Ax, cmap='gray',vmax=np.percentile(Ax ,99))

            AHk = np.transpose(AHk[0,cha,].abs().cpu().numpy(),(1,0))
            plt.imsave(f'{qcdir}AHk.png', AHk,cmap='gray',vmax=np.percentile(AHk ,99))
        
            AHAx = np.transpose(AHAx[0,cha,].abs().cpu().numpy(),(1,0))
            plt.imsave(f'{qcdir}AHAx.png',AHAx,cmap='gray',vmax=np.percentile(AHAx ,99))

            
            for ee in [0]: #range(self.etl):
                img = torch.reshape((traj * torch.permute(inputs[2], [0,2,1])).squeeze(), [inputs[1].shape[-1],etl,-1])[:,ee,]
                img = img.detach().cpu().numpy()
                plt.plot(np.real(img), np.imag(img))
                plt.savefig(f'{qcdir}traj.png')
                plt.clf()

                img = dcf.detach().abs().cpu().numpy()
                plt.imsave(f'{qcdir}dcf.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
            

            if True: # idx == 0:
                pdb.set_trace() # pause for debugging

        return loss

