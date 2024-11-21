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

from models.UnrolledNetwork import UnrolledNetwork
from models.final_layers import *
from utils.utils import *

from time import time

import h5py
from models.UnrolledNetwork import *

class RADTSEUnrolledNetwork(UnrolledNetwork):
    # initialize the required layers
    def __init__(self, A, AH, config, criterion=None):
        ## get all the old good stuff
        super().__init__(A=A, AH=AH, config=config, criterion=criterion)
        
    # test with trained model
    def test_model(self, test_generator, out_dir, indices=None, save_prefix='',
                    load_model_path=None, NUM_EPOCHS=-1, TEs=None,
                   dictionary_path=None, save_t2=False, save_pc=True, save_gif=True, save_kspace=False,
                   vmax=99, roi_path=None, h5_dir=None, ref_echo=0, ema_model=None, fov=320,
                   signal_threshold=False):

        if not os.path.exists(f'{out_dir}images'):
            os.makedirs(f'{out_dir}images')
        # mode model to device
        print(f'\nTEST USING GPU: {torch.cuda.is_available()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        if ema_model is not None:
            ema_model = ema_model.to(device)
            
        # load old model if path is provided
        if load_model_path is not None:
            # load old states
            checkpoint = torch.load(load_model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            NUM_EPOCHS = len(checkpoint['history']['train_loss'])
            if ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        # print DC parameter weights
        tmp = [p.item() for p in self.dc.parameters()]
        for ii in range(len(tmp)):
            print(f'DC parameter weight {ii} = {tmp[ii]:.4f}')

        save_prefix = f'{save_prefix}_{NUM_EPOCHS}epochs' # add num epochs to save prefix
        
        self.eval() # turn dropout off by putting model in train mode
        if ema_model is not None:
            ema_model.eval()

        # figure out which part of anatomy ROIs are for
        if roi_path is not None:
            if 'brain' in roi_path.lower():
                roi_list = ['white_matter','gray_matter','csf','cerebellum']
                anatomy = 'brain'
            elif 'leg' in roi_path.lower():
                roi_list = ['muscle']
                anatomy = 'leg'
            else:
                roi_list = ['liver','Rkidney','Lkidney','spleen','muscle']
                anatomy = 'abdomen'
                
        # start roi stats if available
        if roi_path is not None:
            if not os.path.exists(f'{out_dir}roi_stats'):
                os.makedirs(f'{out_dir}roi_stats')
            if not os.path.exists(f'{out_dir}roi_qc'):
                os.makedirs(f'{out_dir}roi_qc')
            roi_stats_file = f'{out_dir}roi_stats/{save_prefix}_RE{ref_echo:02d}_roi_stats.csv'
            with open(roi_stats_file,'w') as fid:
                print('test_idx', end=',', file=fid)
                for img in ['input','output','reference']:
                    for roi in roi_list:
                        print(f'{img}_{roi}_mean,{img}_{roi}_std,{img}_{roi}_size', end=',', file=fid)
                print('', file=fid)
                
        # do same for t2
        if save_t2 and roi_path is not None:
            if not os.path.exists(f'{out_dir}roi_stats'):
                os.makedirs(f'{out_dir}roi_stats')
            if not os.path.exists(f'{out_dir}roi_qc'):
                os.makedirs(f'{out_dir}roi_qc')
            roi_stats_file_t2 = f'{out_dir}roi_stats/{save_prefix}_RE{ref_echo:02d}_roi_stats_t2.csv'
            with open(roi_stats_file_t2,'w') as fid:
                print('test_idx', end=',', file=fid)
                for img in ['input','output','reference']:
                    for roi in roi_list:
                        print(f'{img}_{roi}_mean,{img}_{roi}_std,{img}_{roi}_size', end=',', file=fid)
                print('', file=fid)

        # determine which samples from the generator to test
        if indices is None:
            indices = range(len(test_generator)) # do all if not declared
        # loop through slices
        for testidx in indices:
            # skip if invalid index was provided
            if testidx > len(test_generator):
                continue

            inputs, targets, prescan_norm = test_generator.__getitem__(testidx)
            R = inputs[2].numel() / inputs[2].sum() # acceleration
            # print progress
            print(f'Testing image {testidx} of {len(test_generator)}')
            print(f'R = {R:.2f}')
            
            # start timer
            tstart = time()

            # make prediction
            with torch.set_grad_enabled(False):
                if ema_model is not None:
                    pred = ema_model(inputs).cpu().detach()
                else:
                    pred = self(inputs).cpu().detach() 
            print(f'Prediction in {time() - tstart:.2f}s')

            # move back to cpu for plotting, apply prescan norm
            for idx,inp in enumerate(inputs):
                inputs[idx] = inp.cpu()
            if targets is not None:
                targets = targets.cpu() * prescan_norm
            inputs[0] = inputs[0] * prescan_norm
            pred = pred * prescan_norm

            # save to h5 file if path is provided
            if h5_dir is not None:
                print('saving h5')
                # save h5 version of output
                if not os.path.exists(f'{h5_dir}{save_prefix}_h5_results/'):
                    os.makedirs(f'{h5_dir}{save_prefix}_h5_results/')
                h5file = h5py.File(f'{h5_dir}{save_prefix}_h5_results/pcmaps_{testidx:03d}.h5', 'w')
                h5file.create_dataset('noisy', data=inputs[0].numpy())
                h5file.create_dataset('recon', data=pred.numpy())
                h5file.create_dataset('basis', data=inputs[-1].numpy())
                if targets is not None:
                    h5file.create_dataset('target', data=targets.numpy())
                h5file.close()
                        

            icase = 0  # batch size = 1
            [_,_,nx,ny] = inputs[0].shape # get img dims

            # FoV for output images
            [_,_,nx,ny] = pred.shape
            xmin = nx // 2 - fov//2
            xmax = nx // 2 + fov//2
            ymin = ny // 2 - fov//2
            ymax = ny // 2 + fov//2

            
            # override for creating figures
            nnx = 208 # even number
            nny = 160 # even number
            xmin = nx // 2 - nnx // 2
            xmax = nx // 2 + nnx // 2
            ymin = ny // 2 - nny // 2
            ymax = ny // 2 + nny // 2
            

            # save PC images
            if save_pc:
                for cha in range(self.ncha):
                    noisy_mag = np.transpose(np.squeeze(inputs[0][icase,cha,xmin:xmax,ymin:ymax].abs().numpy()),(1,0))
                    recon_mag = np.transpose(np.squeeze(pred[icase,cha,xmin:xmax,ymin:ymax].abs().numpy()),(1,0))
                    if targets is not None:
                        target_mag = np.transpose(np.squeeze(targets[icase,cha,xmin:xmax,ymin:ymax].abs().numpy()),(1,0))
                    """
                    noisy_phase = np.transpose(np.squeeze(inputs[0][icase,cha,xmin:xmax,ymin:ymax].angle().numpy()),(1,0))
                    recon_phase = np.transpose(np.squeeze(pred[icase,cha,xmin:xmax,ymin:ymax].angle().numpy()),(1,0))
                    target_phase = np.transpose(np.squeeze(targets[icase,cha,xmin:xmax,ymin:ymax].angle().numpy()),(1,0))
                    """

                    # save
                    plt.imsave(f'{out_dir}images/{save_prefix}_noisymag_{testidx:03d}_cha{cha:02d}.tiff',
                               noisy_mag, cmap='gray', vmin=0, vmax=np.percentile(noisy_mag,vmax))
                    plt.imsave(f'{out_dir}images/{save_prefix}_reconmag_{testidx:03d}_cha{cha:02d}.tiff',
                               recon_mag, cmap='gray', vmin=0, vmax=np.percentile(recon_mag,vmax))
                    if targets is not None:
                        plt.imsave(f'{out_dir}images/{save_prefix}_targetmag_{testidx:03d}_cha{cha:02d}.tiff',
                                   target_mag, cmap='gray', vmin=0, vmax=np.percentile(target_mag,vmax))

                    if save_kspace:
                        # save k space for qc purposes
                        if not os.path.exists(f'{out_dir}kspace/'):
                            os.makedirs(f'{out_dir}kspace/')
                        noisy_k = np.transpose(np.squeeze(fftnc_torch(inputs[0][icase,cha,:,:]).abs().numpy()),(1,0))
                        recon_k = np.transpose(np.squeeze(fftnc_torch(pred[icase,cha,:,:]).abs().numpy()),(1,0))
                        plt.imsave(f'{out_dir}kspace/{save_prefix}_noisyk_{testidx:03d}_cha{cha:02d}.tiff',
                                   noisy_k, cmap='gray', vmin=0, vmax=np.percentile(noisy_mag,vmax))
                        plt.imsave(f'{out_dir}kspace/{save_prefix}_reconk_{testidx:03d}_cha{cha:02d}.tiff',
                                   recon_k, cmap='gray', vmin=0, vmax=np.percentile(recon_mag,vmax))
                        if targets is not None:
                            target_k = np.transpose(np.squeeze(fftnc_torch(targets[icase,cha,:,:]).abs().numpy()),(1,0)) 
                            plt.imsave(f'{out_dir}kspace/{save_prefix}_targetk_{testidx:03d}_cha{cha:02d}.tiff',
                                       target_k, cmap='gray', vmin=0, vmax=np.percentile(target_mag,vmax))

                
                # print ROI stats if there are ROIs available
                if roi_path is not None:
                    if targets is not None:
                        img_list = [noisy_mag, recon_mag, target_mag]
                    else:
                        img_list = [noisy_mag, recon_mag]
                    rois = load_labels(roi_path, testidx, img_dims=[nx,ny], anatomy=anatomy) # load ROIs
                    with open(roi_stats_file, 'a') as fid:
                        print(f'{testidx}', end=',', file=fid)
                        for img in img_list:
                            for roi in roi_list:
                                roiidx = np.transpose(rois[roi][xmin:xmax, ymin:ymax], (1,0))
                                roimean = np.mean(img[roiidx]) # roi mean
                                roistd = np.std(img[roiidx]) # roi standard deviation
                                roisize = np.sum(roiidx) # roi size
                                print(f'{roimean},{roistd},{roisize}', end=',', file=fid)
                        print('', file=fid)
                        
                    # save QC images for ROI placement
                    if targets is not None:
                        qc_mag = target_mag
                    else:
                        qc_mag = noisy_mag
                    qc_max = np.percentile(qc_mag, vmax)
                    plt.clf()
                    plt.imshow(qc_mag, cmap='gray', vmin=0, vmax=qc_max)
                    for roi in roi_list:
                        if rois[roi].sum():
                            mask = np.transpose(rois[roi][xmin:xmax, ymin:ymax],(1,0))
                            plt.imshow(mask, cmap='jet', alpha=0.75 * (mask>0))
                    plt.savefig(f'{out_dir}roi_qc/{save_prefix}_{testidx:03d}.png')
                    plt.clf()

            # save TE images--dictionary file is required for this
            if TEs is not None and len(TEs):
                print('Saving TE images')
                ## convert to TEs
                D = inputs[-1].squeeze()
                # get correct dtype
                if torch.is_tensor(D):
                    D = D.to(inputs[0].device, dtype=torch.complex64)
                else:
                    D = torch.tensor(D, dtype=torch.complex64).to(inputs[0].device)
                # temporal dims
                etl, pc = D.shape
                # go to TE space
                inp_TEs = PC2Contrast(inputs[0][icase,:,xmin:xmax,ymin:ymax], D)
                pred_TEs = PC2Contrast(pred[icase,:,xmin:xmax,ymin:ymax], D)
                if targets is not None:
                    tar_TEs = PC2Contrast(targets[icase,:,xmin:xmax,ymin:ymax], D)

                isVFA = False # default is CFA
                dictidx = testidx // 28 # index of dictionary within h5 file
                D = load_dictionary(dictionary_path, dictidx) # load dictionary
                if np.unique(D['alpha']).shape[0] > 2:
                    isVFA = True # change to VFA if dictionary indicates so

                # save gif of all TEs
                if save_gif:
                    self.save_gif(np.transpose(np.squeeze(inp_TEs.abs().numpy()),(0,2,1)), f'{out_dir}images/{save_prefix}_noisymag_{testidx:03d}.gif', vmax)
                    self.save_gif(np.transpose(np.squeeze(pred_TEs.abs().numpy()),(0,2,1)), f'{out_dir}images/{save_prefix}_reconmag_{testidx:03d}.gif', vmax)
                    if targets is not None:
                        self.save_gif(np.transpose(np.squeeze(tar_TEs.abs().numpy()),(0,2,1)), f'{out_dir}images/{save_prefix}_targetmag_{testidx:03d}.gif', vmax)

                # save each TE requested
                for TE in TEs:
                    noisy_mag = np.transpose(np.squeeze(inp_TEs[TE].abs().numpy()),(1,0))
                    recon_mag = np.transpose(np.squeeze(pred_TEs[TE].abs().numpy()),(1,0))
                    if targets is not None:
                        target_mag = np.transpose(np.squeeze(tar_TEs[TE].abs().numpy()),(1,0))

                    # for CFA, TE equivalent is just the TE
                    TE_eq = np.round(D['TE'] * (TE+1) * 1000).astype(int)
                    
                    # for VFA, we can estimate it using liver ROI if available
                    if isVFA and roi_path is not None and anatomy=='abdomen':
                        rois = load_labels(roi_path, testidx, img_dims=[nx,ny], anatomy=anatomy)
                        liver_roi = rois['liver'][xmin:xmax,ymin:ymax]
                        if liver_roi.sum():
                            sn = tar_TEs[:,liver_roi].abs().numpy()
                            sn = np.mean(sn, axis=1)
                            sn = sn / np.max(sn)
                            fcoh = D['fcoherence'] / np.max(D['fcoherence'])

                            TE_eq = np.round(-45 * np.log(sn[TE] / fcoh[TE])).astype(int)
                            
                    # now save the images with TE in the output name
                    if targets is not None:
                        plt.imsave(f'{out_dir}images/{save_prefix}_targetmag_{testidx:03d}_TE{TE:02d}_{TE_eq}ms.tiff',
                                   target_mag, cmap='gray', vmin=0, vmax=np.percentile(target_mag, vmax))
                        
                    plt.imsave(f'{out_dir}images/{save_prefix}_noisymag_{testidx:03d}_TE{TE:02d}_{TE_eq}ms.tiff',
                               noisy_mag, cmap='gray', vmin=0, vmax=np.percentile(noisy_mag,vmax))
                    plt.imsave(f'{out_dir}images/{save_prefix}_reconmag_{testidx:03d}_TE{TE:02d}_{TE_eq}ms.tiff',
                               recon_mag, cmap='gray', vmin=0, vmax=np.percentile(recon_mag,vmax))
                        
            ## save T2 maps
            if save_t2:
                print('Saving T2 Maps')
                dictidx = testidx // 28
                x_range = [xmin,xmax]
                y_range = [ymin,ymax]
                D = load_dictionary(dictionary_path, dictidx) # load dictionary
                
                # body mask
                body_roi = torch.ones_like(inputs[0][icase,])
                if roi_path is not None:
                    rois = load_labels(roi_path, testidx, img_dims=[nx,ny], anatomy=anatomy) # load ROIs
                    if 'body' in rois.keys():
                        body_roi = torch.from_numpy(rois['body']).to(inputs[0].device)
                    
                # generate T2 maps
                input_T2map, input_fit = gen_t2_maps(inputs[0][icase,]*body_roi, D, x_range, y_range)
                recon_T2map, recon_fit = gen_t2_maps(pred[icase,]*body_roi, D, x_range, y_range)
                if targets is not None:
                    target_T2map, target_fit = gen_t2_maps(targets[icase,]*body_roi, D, x_range, y_range)
                    T2_error = np.abs(recon_T2map - target_T2map) # T2 error if target is provided
                    img_list = [input_T2map, recon_T2map, target_T2map]
                else:
                    img_list = [input_T2map, recon_T2map]

                # print T2 ROI stats if there are ROIs available
                if roi_path is not None:
                    rois = load_labels(roi_path, testidx, img_dims=[nx,ny], anatomy=anatomy) # load ROIs
                    with open(roi_stats_file_t2, 'a') as fid:
                        print(f'{testidx}', end=',', file=fid)
                        for img in img_list:
                            for roi in roi_list:
                                roiidx = rois[roi][xmin:xmax, ymin:ymax]
                                roimean = np.mean(img[roiidx]) # roi mean
                                roistd = np.std(img[roiidx]) # roi standard deviation
                                roisize = np.sum(roiidx) # roi size
                                print(f'{roimean},{roistd},{roisize}', end=',', file=fid)
                        print('', file=fid)

                    # save QC images for ROI placement
                    if targets is not None:
                        qc_mag = np.transpose(np.squeeze(targets[icase,0,xmin:xmax,ymin:ymax].abs().numpy()),(1,0))
                    else:
                        qc_mag = np.transpose(np.squeeze(inputs[0][icase,0,xmin:xmax,ymin:ymax].abs().numpy()),(1,0))
                    qc_max = np.percentile(qc_mag, vmax)
                    plt.clf()
                    plt.imshow(qc_mag, cmap='gray', vmin=0, vmax=qc_max)
                    for roi in roi_list:
                        if rois[roi].sum():
                            mask = np.transpose(rois[roi][xmin:xmax, ymin:ymax],(1,0))
                            plt.imshow(mask, cmap='jet', alpha=0.75 * (mask>0))
                    plt.savefig(f'{out_dir}roi_qc/{save_prefix}_{testidx:03d}.png')
                    plt.clf()

                # threshold out pixels
                T2min = 10
                T2max = 160
                    
                # signal mask
                if signal_threshold:
                    signal_vmax = 0.1
                    t2_vmax = T2max*2
                    noisy_mag = inputs[0][icase,0,xmin:xmax,ymin:ymax].abs()
                    noisy_mask = (noisy_mag > (noisy_mag.max() * signal_vmax)) * (input_T2map < t2_vmax)
                    input_T2map = input_T2map * noisy_mask.cpu().numpy() + (1-noisy_mask.cpu().numpy()) * T2min
                    input_T2map = input_T2map * np.transpose(body_roi[xmin:xmax,ymin:ymax].cpu().numpy(), (0,1))
                    
                    recon_mag = pred[icase,0,xmin:xmax,ymin:ymax].abs()
                    recon_mask = (recon_mag > (recon_mag.max() * signal_vmax)) * (recon_T2map < t2_vmax)
                    recon_T2map = recon_T2map * recon_mask.cpu().numpy() + (1-recon_mask.cpu().numpy()) * T2min
                    recon_T2map = recon_T2map * np.transpose(body_roi[xmin:xmax,ymin:ymax].cpu().numpy(), (0,1))
                    
                    if targets is not None:
                        target_mag = targets[icase,0,xmin:xmax,ymin:ymax].abs()
                        target_mask = (target_mag > (target_mag.max() * signal_vmax)) * (target_T2map < t2_vmax)
                        target_T2map = target_T2map * target_mask.cpu().numpy() + (1-target_mask.cpu().numpy()) * T2min
                        target_T2map = target_T2map * np.transpose(body_roi[xmin:xmax,ymin:ymax].cpu().numpy(), (0,1))


                # Save T2 maps as images
                input_T2map = np.transpose(input_T2map)
                [input_T2map, _] = relaxationColorMap('T2', input_T2map, T2min, T2max)
                recon_T2map = np.transpose(recon_T2map)
                [recon_T2map, T2cmap] = relaxationColorMap('T2', recon_T2map, T2min, T2max)
                if targets is not None:
                    target_T2map = np.transpose(target_T2map)
                    [target_T2map, _] = relaxationColorMap('T2', target_T2map, T2min, T2max)
                    T2_error = np.transpose(T2_error)

                # save as subplot of all T2 maps together
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(input_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)
                plt.title('T2map - Noisy',fontsize=8)
                plt.colorbar()
                plt.axis('off')
                plt.subplot(2,2,2)
                plt.imshow(recon_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)
                plt.title('T2map - Recon',fontsize=8)
                plt.colorbar()
                plt.axis('off')
                if targets is not None:
                    plt.subplot(2,2,3)
                    plt.imshow(target_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)
                    plt.title('T2map - Target',fontsize=8)
                    plt.colorbar()
                    plt.axis('off')
                    plt.subplot(2,2,4)
                    plt.imshow(T2_error, cmap=T2cmap, vmin=0, vmax=T2max/2)
                    plt.title('T2map - Error',fontsize=8)
                    plt.colorbar()
                    plt.axis('off')
                
                plt.savefig(f'{out_dir}images/{save_prefix}_{testidx:03d}_T2.png')
                plt.clf()
                plt.close()
                
                # now save each individually, with colorbar first
                plt.imshow(input_T2map, vmin=T2min, vmax=T2max, cmap=T2cmap)
                plt.colorbar()
                plt.axis('off')
                plt.savefig(f'{out_dir}images/{save_prefix}_noisy_{testidx:03d}_T2_cbar.png', bbox_inches='tight')
                plt.clf()
                plt.close()

                plt.imshow(recon_T2map, vmin=T2min, vmax=T2max, cmap=T2cmap)
                plt.colorbar()
                plt.axis('off')
                plt.savefig(f'{out_dir}images/{save_prefix}_recon_{testidx:03d}_T2_cbar.png', bbox_inches='tight')
                plt.clf()
                plt.close()
                
                
                if targets is not None:
                    plt.imshow(target_T2map, vmin=T2min, vmax=T2max, cmap=T2cmap)
                    plt.colorbar()
                    plt.axis('off')
                    plt.savefig(f'{out_dir}images/{save_prefix}_target_{testidx:03d}_T2_cbar.png', bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    
                    # now without colorbar 
                    plt.imsave(f'{out_dir}images/{save_prefix}_target_{testidx:03d}_T2.tiff',
                               target_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)
                plt.imsave(f'{out_dir}images/{save_prefix}_noisy_{testidx:03d}_T2.tiff',
                           input_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)
                plt.imsave(f'{out_dir}images/{save_prefix}_recon_{testidx:03d}_T2.tiff',
                           recon_T2map, cmap=T2cmap, vmin=T2min, vmax=T2max)

                
                # fit maps
                if targets is not None:
                    target_fit = np.transpose(target_fit.abs().cpu().numpy(), (1,0))
                    plt.imsave(f'{out_dir}images/{save_prefix}_target_{testidx:03d}_T2_fit.tiff',
                               target_fit, cmap='gray', vmin=0, vmax=1)
                    
                input_fit = np.transpose(input_fit.abs().cpu().numpy(), (1,0))
                plt.imsave(f'{out_dir}images/{save_prefix}_noisy_{testidx:03d}_T2_fit.tiff',
                           input_fit>0.9, cmap='gray', vmin=0, vmax=1)
                
                recon_fit = np.transpose(recon_fit.abs().cpu().numpy(), (1,0))
                plt.imsave(f'{out_dir}images/{save_prefix}_recon_{testidx:03d}_T2_fit.tiff',
                           recon_fit>0.9, cmap='gray', vmin=0, vmax=1)
                
                
                # save T2 maps as ouput h5 file if requested
                if h5_dir is not None:
                    print('saving h5')
                    if not os.path.exists(f'{h5_dir}{save_prefix}_h5_results'):
                        os.makedirs(f'{h5_dir}{save_prefix}_h5_results')
                    h5file = h5py.File(f'{h5_dir}{save_prefix}_h5_results/t2maps_{testidx:03d}.h5', 'w')
                    h5file.create_dataset('input_T2', data=input_T2map)
                    h5file.create_dataset('recon_T2', data=recon_T2map)
                    
                    h5file.create_dataset('input_T2_fit', data=input_fit)
                    h5file.create_dataset('recon_T2_fit', data=recon_fit)
                    
                    h5file.create_dataset('body_roi', data=np.transpose(body_roi[xmin:xmax,ymin:ymax].cpu().numpy(), (1,0)))
                    
                    if targets is not None:
                        h5file.create_dataset('target_T2', data=target_T2map)
                        h5file.create_dataset('target_T2_fit', data=target_fit)
                    h5file.close()
                    

    # save gif
    def save_gif(self, img_stack, oname, vmax=99):
        import matplotlib.animation as animation
        plt.clf()
        plt.close()
        fig = plt.figure()
        ims = []
        for TE in range(img_stack.shape[0]):
            M = np.percentile(img_stack[TE,], vmax)
            im = plt.imshow(img_stack[TE,], cmap='gray', vmin=0, vmax=M)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500)
        ani.save(oname, writer=animation.PillowWriter())
        plt.clf()
        plt.close()

        
    def validate_t2(self, valid_generator, out_dir, save_prefix='',
                    load_model_path=None, dictionary_path=None, roi_path=None,
                    ema_model=None):
        ''' This method is used to validate the T2 error on a validation set for model selection '''

        out_dir = f'{out_dir}t2val' # output directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # mode model to device
        print(f'\nTEST USING GPU: {torch.cuda.is_available()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        if ema_model is not None:
            ema_model = ema_model.to(device)

        # load old model if path is provided
        if load_model_path is not None:
            # load old states
            checkpoint = torch.load(load_model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            NUM_EPOCHS = len(checkpoint['history']['train_loss'])
            if ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                
        else:
            NUM_EPOCHS = -1

        # print DC parameter weights
        tmp = [p.item() for p in self.dc.parameters()]
        for ii in range(len(tmp)):
            print(f'DC parameter weight {ii} = {tmp[ii]:.4f}')
        
        save_prefix = f'{save_prefix}_{NUM_EPOCHS}epochs'
        
        self.eval() # turn dropout off by putting model in train mode
        if ema_model is not None:
            ema_model.eval()
        
        # figure out which part of anatomy ROIs are for
        if roi_path is not None:
            if 'brain' in roi_path.lower():
                roi_list = ['white_matter','gray_matter','csf','cerebellum']
                anatomy = 'brain'
            elif 'leg' in roi_path.lower():
                roi_list = ['muscle']
                anatomy = 'leg'
            else:
                roi_list = ['liver','Rkidney','Lkidney','spleen','muscle']
                anatomy = 'abdomen'

        # begin output CSV
        with open(f'{out_dir}/t2_error_{NUM_EPOCHS}epochs.csv', 'w') as fid:
            fid.write('slice,pixels_per_slice,body_size,body_l1,body_l2,')
            for roi in roi_list:
                fid.write(f'{roi}_size,{roi}_l1,{roi}_l2,')
            fid.write('\n')
        
        for testidx in range(len(valid_generator)):
            if testidx > len(valid_generator):
                continue

            # get items
            inputs, targets, _, = valid_generator.__getitem__(testidx)
            R = inputs[2].numel() / inputs[2].sum()
            # print progress
            print(f'Testing image {testidx} of {len(test_generator)}')
            print(f'R = {R:.2f}')
            
            # start timer
            tstart = time()

            # make prediction
            with torch.set_grad_enabled(False):
                if ema_model is not None:
                    pred = ema_model(inputs).cpu().detach()
                else:
                    pred = self(inputs).cpu().detach() 
            print(f'Prediction in {time() - tstart:.2f}s')
                
            # move back to cpu for calculations with ROI
            for idx,inp in enumerate(inputs):
                inputs[idx] = inp.cpu()
            if targets is not None:
                targets = targets.cpu()
                
            icase = 0  # batch_size = 1
            # cropping info
            [_,_,nx,ny] = inputs[0].shape
            nnx = 224 # even number
            nny = 160 # even number
            xmin = nx // 2 - nnx // 2
            xmax = nx // 2 + nnx // 2
            ymin = ny // 2 - nny // 2
            ymax = ny // 2 + nny // 2

            # fit T2 maps
            tstart = time()
            dictidx = testidx // 28
            D = load_dictionary(dictionary_path, dictidx)
            recon_T2map, recon_fit = gen_t2_maps(pred[icase,], D, [xmin,xmax], [ymin,ymax])
            target_T2map, target_fit = gen_t2_maps(targets[icase,], D, [xmin,xmax], [ymin,ymax])
            print(f'T2maps in {time() - tstart:.2f}s')

            # error map
            diff = target_T2map - recon_T2map

            # crude signal threshold roi 
            thresh = torch.quantile(targets[icase,0,xmin:xmax,ymin:ymax].abs(), 0.75)
            sig_roi = np.transpose((targets[icase,0,xmin:xmax,ymin:ymax].abs() > thresh).numpy().astype(int), [1,0])

            # l1 and l2 error on signal roi
            l1 = np.linalg.norm((diff * sig_roi).flatten(),1) / np.sum(sig_roi)
            l2 = np.linalg.norm((diff * sig_roi).flatten(),2) / np.sum(sig_roi)

            # print 
            with open(f'{out_dir}/t2_error_{NUM_EPOCHS}epochs.csv', 'a') as fid:
                fid.write(f'{testidx},{np.size(diff)},{np.sum(sig_roi)},{l1},{l2},')

            # pre-computed rois
            rois = load_labels(roi_path, testidx, img_dims=[nx,ny], anatomy=anatomy)
            for roi in roi_list:
                sig_roi = rois[roi][xmin:xmax, ymin:ymax].astype(int)
                if np.sum(sig_roi) == 0:
                    l1 = np.nan
                    l2 = np.nan
                else:
                    l1 = np.linalg.norm((diff * sig_roi).flatten(),1) / np.sum(sig_roi)
                    l2 = np.linalg.norm((diff * sig_roi).flatten(),2) / np.sum(sig_roi)
            
                with open(f'{out_dir}/t2_error_{NUM_EPOCHS}epochs.csv', 'a') as fid:
                    fid.write(f'{np.sum(sig_roi)},{l1},{l2},')

                    
            with open(f'{out_dir}/t2_error_{NUM_EPOCHS}epochs.csv', 'a') as fid:
                fid.write(f'\n')


    
    # test with trained model
    def error_metrics(self, test_generator, out_dir, indices=None, save_prefix='',
                    load_model_path=None, NUM_EPOCHS=-1, ema_model=None, mag=True,
                    TEs=[], q=1, fov=256, vmax=100):
        # mode model to device
        print(f'\nTEST USING GPU: {torch.cuda.is_available()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        if ema_model is not None:
            ema_model = ema_model.to(device)

        if TEs is None:
            TEs = [0]

        # make directory if not already made
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # load old model if path is provided
        if load_model_path is not None:
            # load old states
            checkpoint = torch.load(load_model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            NUM_EPOCHS = len(checkpoint['history']['train_loss'])
            if ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                
        qstr = f'{q}'.replace('.','p')
        save_prefix = f'{save_prefix}_{NUM_EPOCHS}epochs_q{qstr}'
        
        self.eval() # turn dropout off by putting model in train mode    

        # determine which samples from the generator to test
        if indices is None:
            indices = range(len(test_generator)) # do all if not declared


        if test_generator.image_type == 'pcs':
            methods = ['inp','dl','llr','nlr3d']
        else:
            methods = ['inp','dl']
        
        with open(f'{out_dir}/{save_prefix}_metrics.csv', 'w') as fid:
            fid.write('slice,pixels_per_slice,')
            for m in methods:
                fid.write(f'{m}_l1,{m}_l2,{m}_psnr,')
            fid.write('\n')
            
        # initialize metrics
        mean_l1 = {}
        mean_l2 = {}
        mean_psnr = {}
        for m in methods:
            mean_l1[m] = 0
            mean_l2[m] = 0
            mean_psnr[m] = 0
        
        # loop through slices
        for testidx in indices:
            print(f'slice {testidx} of {len(test_generator)}')
            
            # skip if invalid index was provided
            if testidx > len(test_generator):
                continue

            inputs, targets, prescan_norm = test_generator.__getitem__(testidx)
            
            if test_generator.image_type == 'pcs':
                traj, dcf, D = test_generator.__get_traj_dcf_dict__()
                llr = test_generator.__get_cs__(testidx, 'llr').to(device)
                nlr3d = test_generator.__get_cs__(testidx, 'nlr3d').to(device)

            # move input to device
            for idx,inp in enumerate(inputs):
                inputs[idx] = inp.to(device)
            if targets is not None:
                targets = targets.to(device)
            prescan_norm = prescan_norm.to(device)
            
            # make prediction
            with torch.set_grad_enabled(False):
                if ema_model is not None:
                    pred = ema_model(inputs).detach()
                else:
                    pred = self(inputs).detach()

            # convert from PCs to TEs
            if test_generator.image_type == 'pcs':
                ## convert to TEs
                D = inputs[-1].squeeze()
                # get correct dtype
                if torch.is_tensor(D):
                    D = D.to(device, dtype=torch.complex64)
                else:
                    D = torch.tensor(D, dtype=torch.complex64).to(device)
                # temporal dims
                etl, pc = D.shape
                # go to TE space
                inputs[0] = PC2Contrast(inputs[0][0,], D).unsqueeze(0)
                pred = PC2Contrast(pred[0,], D).unsqueeze(0)
                if targets is not None:
                    targets = PC2Contrast(targets[0,], D).unsqueeze(0)
                llr = PC2Contrast(llr[0,], D).unsqueeze(0)
                nlr3d = PC2Contrast(nlr3d[0,], D).unsqueeze(0)
                
                llr = llr * prescan_norm
                nlr3d = nlr3d * prescan_norm

            # apply prescan norm--only affects qc images, not error
            pred = pred * prescan_norm
            inputs[0] = inputs[0] * prescan_norm
            if targets is not None:
                targets = targets * prescan_norm

            
            # qc directory
            qcdir = f'{out_dir}/error_qc_q{qstr}/'
            if not os.path.exists(qcdir):
                os.makedirs(qcdir)
            R = torch.numel(inputs[2]) / inputs[2].sum()
            histmax = 0.25

            # FoV for error
            [_,_,nx,ny] = pred.shape
            xmin = nx // 2 - fov//2
            xmax = nx // 2 + fov//2
            ymin = ny // 2 - fov//2
            ymax = ny // 2 + fov//2

            # crop images
            pred = pred[:,:,xmin:xmax,ymin:ymax]
            inputs[0] = inputs[0][:,:,xmin:xmax,ymin:ymax]
            targets = targets[:,:,xmin:xmax,ymin:ymax,]
            if test_generator.image_type == 'pcs':
                llr = llr[:,:,xmin:xmax,ymin:ymax]
                nlr3d = nlr3d[:,:,xmin:xmax,ymin:ymax]
            
            # normalize
            if q == -1: # normalize by mean
                pred = pred / pred.abs().mean()
                inputs[0] = inputs[0] / inputs[0].abs().mean()
                targets = targets / targets.abs().mean()
                if test_generator.image_type == 'pcs':
                    llr = llr / llr.abs().mean()
                    nlr3d = nlr3d / nlr3d.abs().mean()
            elif q > 0: # normalize by quantile
                pred = pred / torch.quantile(pred.abs(), q)
                inputs[0] = inputs[0] / torch.quantile(inputs[0].abs(), q)
                targets = targets / torch.quantile(targets.abs(), q)
                if test_generator.image_type == 'pcs':
                    llr = llr / torch.quantile(llr.abs(), q)
                    nlr3d = nlr3d / torch.quantile(nlr3d.abs(), q)

            # now re-normalize all by the target max
            pred = pred / targets.abs().max()
            inputs[0] = inputs[0] / targets.abs().max()
            if test_generator.image_type == 'pcs':
                llr = llr / targets.abs().max()
                nlr3d = nlr3d / targets.abs().max()
            targets = targets / targets.abs().max()
            

            if mag:
                pred = pred.abs()
                targets=targets.abs()
                inputs[0] = inputs[0].abs()
                if test_generator.image_type == 'pcs':
                    llr = llr.abs()
                    nlr3d = nlr3d.abs()
            
            if test_generator.image_type == 'pcs':
                recons = [inputs[0], pred, llr, nlr3d]
            else:
                recons = [inputs[0], pred]
            # calculate metrics for this slice
            l1_slc = {}
            l2_slc = {}
            psnr_slc = {}

            for m, recon in zip(methods, recons):
                zero_slc = torch.zeros_like(targets)
                l1_slc[m] = mae(targets, recon) / mae(targets, zero_slc) 
                l2_slc[m] = torch.sqrt(mse(targets, recon)) / torch.sqrt(mse(targets, zero_slc))
                psnr_slc[m] = psnr(targets.abs(), recon.abs())

                mean_l1[m] += l1_slc[m] / len(indices)
                mean_l2[m] += l2_slc[m] / len(indices)
                mean_psnr[m] += psnr_slc[m] / len(indices)

            # plot qc images
            for TE in TEs:
                
                # extra cropping for qc images -- for image generation only, not error
                [_,_,nx,ny] = targets.shape # get img dims
            
                nnx = 224 # 208 # even number
                nny = 160 # even number
                xmin = nx // 2 - nnx // 2
                xmax = nx // 2 + nnx // 2
                ymin = ny // 2 - nny // 2
                ymax = ny // 2 + nny // 2

                # qc image for target
                tar = np.transpose(targets[0,TE,xmin:xmax,ymin:ymax].abs().cpu().numpy(), (1,0))
                temax = np.percentile(tar, vmax)
                plt.imsave(f'{qcdir}/TAR_{testidx:03d}_TE{TE:02d}.tiff', tar, cmap='gray', vmax=temax) 
                plt.imshow(tar, cmap='gray', vmax=temax)
                plt.colorbar()
                plt.savefig(f'{qcdir}/TAR_{testidx:03d}_TE{TE:02d}_cbar.tiff')
                plt.close()
                
                for m, recon in zip(methods,recons):
                    # save image
                    img = np.transpose(recon[0,TE,xmin:xmax,ymin:ymax].abs().cpu().numpy(), (1,0))
                    plt.imsave(f'{qcdir}/{m.upper()}_{testidx:03d}_TE{TE:02d}.tiff', img, cmap='gray', vmax=temax)

                    # produce error map
                    tar = np.transpose(targets[0,TE,xmin:xmax,ymin:ymax].abs().cpu().numpy(), (1,0))
                    error = np.abs(img - tar)
                    
                    # save error map with and without cbar
                    plt.imsave(f'{qcdir}/ERROR_{m.upper()}_{testidx:03d}_TE{TE:02d}.tiff', error, cmap='gray', vmax=temax)
                    plt.imshow(error, cmap='gray', vmax=temax)
                    plt.colorbar()
                    plt.savefig(f'{qcdir}/ERROR_{m.upper()}_{testidx:03d}_TE{TE:02d}_cbar.tiff')
                    plt.close()

            # print slice metrics to csv
            with open(f'{out_dir}/{save_prefix}_metrics.csv', 'a') as fid:
                fid.write(f'{testidx},{torch.numel(targets)},')
                for m in methods:
                    fid.write(f'{l1_slc[m]},{l2_slc[m]},{psnr_slc[m]},')
                fid.write('\n')


        with open(f'{out_dir}/{save_prefix}_metrics.csv', 'a') as fid:
            fid.write(f'MEAN,,')
            for m in methods:
                fid.write(f'{mean_l1[m]},{mean_l2[m]},{mean_psnr[m]},')
            fid.write('\n')


