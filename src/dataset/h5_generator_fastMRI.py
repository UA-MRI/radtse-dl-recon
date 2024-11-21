import numpy as np
import h5py as h5py
import scipy.stats as scs
import os
from PIL import Image

# pytorch
import torch
from torch.utils.data import Dataset

# my stuff
from utils.utils import *

# debugging
import pdb
import matplotlib.pyplot as plt
from time import time

def pad_crop(img, dim):
    if img.shape[-2] < dim[0]:
        xpad = dim[0] - img.shape[-2]
        xpad = (np.floor(xpad/2).astype(int), np.ceil(xpad/2).astype(int))
        img = np.pad(img,((0,0), (0,0),  xpad, (0,0)))
    if img.shape[-1] < dim[1]:
        ypad = dim[1] - img.shape[-1]
        ypad = (np.floor(ypad/2).astype(int), np.ceil(ypad/2).astype(int))
        img = np.pad(img,((0,0), (0,0),  (0,0), ypad))
    xmin = np.round(img.shape[-2] / 2 - dim[0] / 2).astype(int)
    xmax = np.round(img.shape[-2] / 2 + dim[0] / 2).astype(int)
    ymin = np.round(img.shape[-1] / 2 - dim[1] / 2).astype(int)
    ymax = np.round(img.shape[-1] / 2 + dim[1] / 2).astype(int)
    return img[:,:,xmin:xmax, ymin:ymax]


def norm(img):
    img = img - img.min()
    return img / img.max()

########################################## FASTMRI ############################################
    
class h5_generator_fastMRI(Dataset):
    'Generates complex-valued data with raw data (for data consistency) for pytorch model'

    def __init__(self, h5_in_path, h5_out_path, traj_type='radial', acceleration=1, accel_type='PI', center=0.1, nlin=320, ncol=512, img_dims=None):
        'Initialization'
        self.dtype = np.complex64
        if not hasattr(self, 'dtype'):
            self.dtype = np.float32
        self.h5_in_path = h5_in_path
        self.h5_out_path = h5_out_path
        
        # traj type. 'cartesian' for cartesian grid, 'radial' for radial nufft
        self.traj_type = traj_type
        self.nlin = nlin # total lines for radial trajectory, unused for cartesian
        self.ncol = ncol # total readout points for radial, unused for cartesian

        self.raw_kspace = self._load_h5_file_with_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        """ 
        #########
        img_dims = None
        # self.raw_kspace = np.expand_dims(self.raw_kspace,1)
        #######
        """
        self.raw_kspace = self.raw_kspace[:,:,::2,:]
        if img_dims[-1] < self.raw_kspace.shape[-1]:
            self.raw_kspace = pad_crop(self.raw_kspace, [img_dims[-1], img_dims[-1]])
        
            
        self.img = ifftnc(self.raw_kspace, axes=(-2,-1))

        
        # save the size of the images
        if img_dims is None:
            self.dim = self.img.shape[2:]
        else:
            self.dim = img_dims
            self.img = pad_crop(self.img, self.dim)

        # get amount of data samples in train/val/test
        self.n_samples = self.img.shape[0]
        self.n_channels = self.img.shape[1]

        # prepare data
        self._prepare_data()
        
        self.acceleration = acceleration
        # data dimensions: kx x ky, only undersample along ky
        self.center = np.floor(center * self.dim[1])  # full sampled center

        # acceleration type: 'PI' = Parallel Imaging, 'CS' = Compressed Sensing
        self.accel_type = accel_type

    def _load_h5_file_with_data(self):
        
        """Method for loading .h5 files
      
        :returns: dict that contains name of the .h5 file as stored in the .h5 file, as well as a generator of the data
        """
        h5file = h5py.File(self.h5_in_path, 'r')
        data = h5file['kspace'][:]

        h5file.close()

        return data
      

    def __len__(self):
        'Denotes the number of samples'
        return int(self.n_samples)


    def _prepare_data(self):
        'Data preparation'
        # normalize magnitude
        self.img = self._normalize(self.img)
        self.img = self.img.astype(self.dtype)

        # generate k-space
        if self.traj_type == 'cartesian':
            self.kspace = self.raw_kspace
        else:
            self.traj, self.dcf = calculate_radial2d_traj_and_dcf(self.ncol, self.nlin)
            self.traj = self.traj.astype(np.float32)
            self.dcf = self.dcf.astype(np.float32)
            self.nufft = FINuFFT(self.traj, self.dim, self.dcf)
            # split it up to ease up on memory
            self.kspace = torch.empty((*self.img.shape[:2], self.nlin, self.ncol), dtype=torch.complex64)
            idxlist = np.floor(np.linspace(0,self.n_samples,min(self.n_samples,10)))
            for i in range(len(idxlist)-1):
                i1, i2 = int(idxlist[i]), int(idxlist[i+1])

                self.kspace[i1:i2,:,:,:] = self.nufft.forward_op(torch.tensor(self.img[i1:i2,:,:,:])).cpu()
            

    def _subsample(self, kspace):
        'Retrospective undersampling/sub-Nyquist sampling'
        if self.traj_type == 'cartesian':
            mask = np.zeros(self.dim, dtype=np.float32)
        else:
            mask = np.zeros((self.nlin,self.ncol), dtype=np.float32)

        # fully sampled center
        if self.traj_type=='cartesian':
            fscenter = (int(np.floor(self.dim[1] / 2 - self.center / 2)), int(np.floor(self.dim[1] / 2 + self.center / 2)))
            mask[:, fscenter[0]:fscenter[1]] = 1

        if self.accel_type == 'PI':
            # Parallel imaging undersampling
            # sample every n-th phase-encoding line
            if self.traj_type == 'cartesian':
                mask[:, ::self.acceleration] = 1
            else:
                mask[::self.acceleration,:] = 1

        elif self.accel_type == 'CS':
            # Compressed Sensing like undersampling
            # calculate amount of points to sample, considering the fully sampled center
            to_sample = np.floor(mask.shape[1] / self.acceleration)
            if self.traj_type=='cartesian':
                nsampled = self.center
            else:
                nsampled = 0
            # effective acceleration rate for high-frequency region. Considering fully sampled center and effective acceleration yields the overall desired acceleration
            #eff_accel = self.dim[1] / (to_sample - self.center)

            # Gaussian sampling
            x = np.arange(0, mask.shape[1])
            # stddev = np.floor(self.dim[1]/4)
            stddev = np.floor(mask.shape[1]/4)
            xU, xL = x + 0.5, x - 0.5
            # prob = scs.norm.cdf(xU, loc=np.floor(self.dim[1]/2), scale=stddev) - scs.norm.cdf(xL, loc=np.floor(self.dim[1]/2), scale=stddev)  # calculate sampling prob. P(xL < x <= xU) = CDF(xU) - CDF(xL), with Gaussian CDF
            prob = scs.norm.cdf(xU, loc=np.floor(mask.shape[1]/2), scale=stddev) - scs.norm.cdf(xL, loc=np.floor(mask.shape[1]/2), scale=stddev)  # calculate sampling prob. P(xL < x <= xU) = CDF(xU) - CDF(xL), with Gaussian CDF
            prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
            while(nsampled < to_sample):
                nums = np.random.choice(x, size=1, p=prob)
                if mask[0, nums] == 0:
                    if self.traj_type=='cartesian':
                        mask[:, nums] = 1
                    else:
                        mask[nums,:] = 1
                    nsampled += 1

        return kspace, mask

    def h5_generate(self):
        'Generates h5 dataset'
        # first delete h5 file if it already exists
        if os.path.exists(self.h5_out_path):
            os.remove(self.h5_out_path)
            
        # Initialization
        target = np.empty((self.n_samples, 1, *self.dim), dtype=self.dtype) # target/reference output
        smaps = np.empty((self.n_samples, self.n_channels, *self.dim), dtype=self.dtype) # coil sensitivity maps
        noisy = np.empty((self.n_samples, 1, *self.dim), dtype=self.dtype) # corrupted/aliased/noisy input
        if self.traj_type == 'cartesian':
            kspace = np.empty((self.n_samples, self.n_channels, *self.dim), dtype=self.dtype) # rawdata/k-space for data consistency
            mask = np.empty((self.n_samples, *self.dim), dtype=np.float32) # sampling mask
        else:
            kspace = np.empty((self.n_samples, self.n_channels, self.nlin, self.ncol), dtype=self.dtype) # rawdata/k-space for data consistency
            mask = np.empty((self.n_samples, self.nlin, self.ncol), dtype=np.float32) # sampling mask

        # Generate data
        for IDX in range(self.n_samples):
            # Get sample
            sample = np.expand_dims(self.img[IDX,],0)


            kspace[IDX,], mask[IDX,] = self._subsample(self.kspace[IDX,])

            # Generate noisy input
            if self.traj_type == 'cartesian':
                noisy_coils = np.fft.ifft2(kspace[IDX,], axes=(1,2))
                # generate coil sensitivity maps
                tstart = time()
                print(f'Computing sensitivy maps for slice {IDX+1} of {self.n_samples}')
                smap = compute_csm(noisy_coils)
                print(f'Calculation completed in {time() - tstart:.2f} seconds')

                
                # noisy[IDX, ...] = torch.sum(noisy_coils * np.conjugate(smap), axis=0)
            else:
                multicoil_data = torch.from_numpy(kspace[IDX,]*mask[IDX,]).to(self.device, torch.complex64).unsqueeze(0)
                noisy_coils = self.nufft.adjoint_op(multicoil_data).cpu()
                # generate coil sensitivity maps
                tstart = time()
                if self.n_channels > 1:
                    print(f'Computing sensitivy maps for slice {IDX+1} of {self.n_samples}')
                    smap = compute_csm(noisy_coils, kSize=8, CalibSize=8)
                    print(f'Calculation completed in {time() - tstart:.2f} seconds')
                else:
                    smap = np.expand_dims(np.ones(smaps[0,].shape),0)
                    
                noisy[IDX, ...] = np.sum(noisy_coils.cpu().numpy() * np.conjugate(smap), axis=1)

            # normalize to the range [0, 1]
            sample = self._normalize(sample)

            # Store sample
            target[IDX, ...] = np.sum(sample * np.conjugate(smap), axis=1)
            smaps[IDX,...] = smap

            

            ####################### QC #########################
            ## QC
            
            if False:
            # if IDX == 0:
                qc_dir = '/clusterscratch/tonerbp/qc/fastMRI_qc_h5gen/'
                if not os.path.exists(qc_dir):
                    os.makedirs(qc_dir)
                base = self.h5_out_path.split('/')[-1].replace('.h5','.tiff')
                print(f'target size: {target.shape}, {target.dtype}')
                qcm = np.abs(target[IDX,0,:,:])
                plt.imsave(f'{qc_dir}qc_target_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                qcm = np.abs(fftnc(target[IDX,0,:,:]))
                plt.imsave(f'{qc_dir}qc_target_k_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                print(f'smaps size: {smaps.shape}, {smaps.dtype}')
                print(f'noisy size: {noisy.shape}, {noisy.dtype}')
                qcm = np.abs(noisy[IDX,0,:,:])
                plt.imsave(f'{qc_dir}qc_noisy_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                qcm = np.abs(fftnc(noisy[IDX,0,:,:]))
                plt.imsave(f'{qc_dir}qc_noisy_k_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                print(f'kspace size: {kspace.shape}, {kspace.dtype}')
                print(f'mask size: {mask.shape}, {mask.dtype}')

                
                inp_error = np.abs(norm(np.abs(noisy[IDX,0,])) - norm(np.abs(target[IDX,0,])))
                plt.imshow(inp_error, cmap='jet', vmin=0, vmax=np.max(inp_error))
                plt.colorbar()
                plt.savefig(f'{qc_dir}inp_error.png')
                plt.clf()
                plt.close()
                
                with h5py.File(self.h5_in_path, 'r') as h5file:
                    pdb.set_trace()
                    tmp = np.abs(h5file['reconstruction_rss'][IDX,])
                    plt.imsave(f'{qc_dir}rss_{base}.tiff', tmp, cmap='gray', vmax=np.percentile(tmp,99))
                    tmp = np.abs(h5file['reconstruction_esc'][IDX,])
                    plt.imsave(f'{qc_dir}esc_{base}.tiff', tmp, cmap='gray', vmax=np.percentile(tmp,99))
        

                """
                qcm = np.abs(smaps[0,0,:,:])
                plt.imsave(f'{qc_dir}qc_smaps_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                
                qcm = np.abs(kspace[0,0,:,:])
                plt.imsave(f'{qc_dir}qc_kspace_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                qcm = np.abs(mask[0,:,:])
                plt.imsave(f'{qc_dir}qc_mask_{base}', qcm, cmap='gray', vmin=0, vmax=np.percentile(qcm,99))
                """

                pdb.set_trace()
            ###################################################

            
        
        with h5py.File(self.h5_out_path, 'w') as h5file:
            h5file.create_dataset('target', data=target)
            h5file.create_dataset('smaps', data=smaps)
            h5file.create_dataset('noisy', data=noisy)
            h5file.create_dataset('kspace', data=kspace)
            h5file.create_dataset('mask', data=mask)

        
        return [noisy, kspace, mask, smaps], target
        
    def _normalize(self, x, min=0, max=1):
        'Normalization'
        # only scale the magnitude
        if np.iscomplexobj(x):
            eps=1e-9
            xabs = np.abs(x)
            normed_magn = (xabs - np.min(xabs)) / (np.max(xabs) - np.min(xabs)) * (max - min) + min
            normed_magn = np.maximum(eps, normed_magn)
            phase =  np.angle(x)
            return normed_magn * np.exp(1j * phase)
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x)) * (max - min) + min

                                            
    
if __name__ == "__main__":
    unittest.main()


