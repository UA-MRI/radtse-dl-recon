import numpy as np
import os
from time import time
import copy
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision


from models.final_layers import *
from merlinth.losses.pairwise_loss import PSNRLoss
# from merlinth.losses.ssim import SSIM as SSIMLoss
from operators.A_functions import *

from utils.utils import *

# Plot loss curve
def plot_loss_curve(train_loss_history, valid_loss_history, out_dir):
    NUM_EPOCHS = len(train_loss_history) # get number of epochs available
    # plot the loss
    fig = plt.figure(figsize=(16,12)) 
    plt.title("Loss vs. Number of Epochs",fontsize=38)
    plt.xlabel("Training Epochs",fontsize=32)
    plt.ylabel("Loss",fontsize=32)
    plt.plot(range(1,NUM_EPOCHS+1),train_loss_history,label="Training Loss")
    plt.plot(range(1,NUM_EPOCHS+1),valid_loss_history,label='Validation Loss')
    # plt.xticks(np.arange(0, NUM_EPOCHS+1, 5.0),fontsize=26)
    plt.xticks(fontsize=26)
    plt.ylim([np.min(np.append(train_loss_history[1:], valid_loss_history[1:])),
              np.max(np.append(train_loss_history[1:], valid_loss_history[1:]))])
    plt.yticks(fontsize=26)
    plt.legend(fontsize=32)
    # save the plot
    if not os.path.exists(f'{out_dir}/plots'):
        os.makedirs(f'{out_dir}/plots')
    plt.savefig(f'{out_dir}/plots/loss_curve.png')
    plt.clf()
    plt.close()
    
# Plot loss curve
def plot_psnr_curve(train_psnr_history, valid_psnr_history, out_dir):
    NUM_EPOCHS = len(train_psnr_history) # get number of epochs available
    # plot the psnr
    fig = plt.figure(figsize=(16,12)) 
    plt.title("PSNR vs. Number of Epochs",fontsize=38)
    plt.xlabel("Training Epochs",fontsize=32)
    plt.ylabel("PSNR",fontsize=32)
    plt.plot(range(1,NUM_EPOCHS+1),train_psnr_history,label="Training PSNR")
    plt.plot(range(1,NUM_EPOCHS+1),valid_psnr_history,label='Validation PSNR')
    # plt.xticks(np.arange(0, NUM_EPOCHS+1, 5.0),fontsize=26)
    plt.xticks(fontsize=26)
    plt.ylim([np.min(np.append(train_psnr_history[0:], valid_psnr_history[0:])),
              np.max(np.append(train_psnr_history[0:], valid_psnr_history[0:]))])
    plt.yticks(fontsize=26)
    plt.legend(fontsize=32)
    # save the plot
    if not os.path.exists(f'{out_dir}/plots'):
        os.makedirs(f'{out_dir}/plots')
    plt.savefig(f'{out_dir}/plots/psnr_curve.png')
    plt.clf()
    plt.close()

# save qc images
def epoch_qc(config, qcdir, yhat, targets, inputs, loss_mask, prefix, sample, prescan_norm, eco=None):
    
    yhat_qc = yhat
    noisy = inputs[0]
    ksp_inp = inputs[1]
    tar = targets
    loss_mask_qc = loss_mask
        
    if not os.path.exists(qcdir):
        os.makedirs(qcdir)
                    
    for cha in [0]: # range(yhat_qc.shape[1]):

        if prescan_norm is None:
            prescan_norm = torch.ones_like(tar[sample,cha,...].cpu())
                            
        img = tar[sample,cha,...].abs().cpu().numpy() * prescan_norm[sample,].abs().numpy()
        plt.imsave(f'{qcdir}{prefix}full_targets.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
        img = (yhat_qc)[sample,cha,...].detach().abs().cpu().numpy()* prescan_norm[sample,].abs().numpy()
        plt.imsave(f'{qcdir}{prefix}yhat.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
        img = (noisy)[sample,cha,...].detach().abs().cpu().numpy()* prescan_norm[sample,].abs().numpy()
        plt.imsave(f'{qcdir}{prefix}input.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))

        # cart kspace
        img = np.abs(fftnc(tar[sample,cha,...].cpu().numpy()))
        plt.imsave(f'{qcdir}{prefix}full_targets_k.tiff', np.transpose(img), cmap='gray', vmax = np.percentile(img, 99))
        img = np.abs(fftnc(yhat_qc[sample,cha,...].detach().cpu().numpy()))
        plt.imsave(f'{qcdir}{prefix}yhat_k.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
        img = np.abs(fftnc(noisy[sample,cha,...].detach().cpu().numpy()))
        plt.imsave(f'{qcdir}{prefix}input_k.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))


        if config['kspace_loss']:
            img = (ksp_inp)[sample,0,...].detach().abs().cpu().numpy()
            plt.imsave(f'{qcdir}{prefix}kspace_full.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
            img = (ksp_inp*inputs[2].unsqueeze(1))[sample,0,...].detach().abs().cpu().numpy()
            plt.imsave(f'{qcdir}{prefix}kspace_input.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))

            img = (ksp_inp*loss_mask_qc)[sample,0,...].detach().abs().cpu().numpy()
            plt.imsave(f'{qcdir}{prefix}kspace_loss.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
                        
            img = inputs[2][sample,:,:].detach().abs().cpu().numpy()
            plt.imsave(f'{qcdir}{prefix}SS_mask.tiff', np.transpose(img), cmap='gray', vmin=0, vmax=1)
                            
            img = loss_mask_qc[sample,0,:,:].detach().abs().cpu().numpy()
            plt.imsave(f'{qcdir}{prefix}loss_mask.tiff', np.transpose(img), cmap='gray', vmin=0, vmax=1)
            
            if config['corner_penalty']:
                img = (yhat_k_cart_qc*corner_mask)[sample,cha,...].detach().abs().cpu().numpy()
                plt.imsave(f'{qcdir}{prefix}yhat_k_cart_mask.tiff', np.transpose(img), cmap='gray', vmax = np.percentile(img, 99))
                img = corner_mask.cpu().numpy()
                plt.imsave(f'{qcdir}{prefix}corner_mask.tiff', np.transpose(img), cmap='gray', vmax=np.percentile(img, 99))
        

# train the model
def train_model(model, optimizer, dataloaders, config, A=None, AH=None,
                save_train_curve=True, tensorboard=False,
                   PSNR=None, SSIM=None, ema_model=None):

    # read config
    NUM_EPOCHS = config['NUM_EPOCHS']
    out_dir = config['odir']
    load_model_path = config['load_model_path']
    parallel = config['parallel']
    GNVN = config['GNVN']
    use_scheduler = config['use_scheduler']
    ncha = config['in_channels'] 
    
    # initialize tensorboard
    if tensorboard:
        writer = SummaryWriter(log_dir = f'{out_dir}/runs/')
    
    print(f'\nTRAIN USING GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if parallel and torch.cuda.device_count() > 1:
        print(f"Distributing over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    if config['ema']:
        ema_model = ema_model.to(device)

    # create psnr and ssim objects
    if PSNR is None:
        PSNR = PSNRLoss()
    
    # load old model if path is provided
    if load_model_path is not None:
        # load old states
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        best_epoch_psnr = checkpoint['best_epoch_psnr']
        valid_loss_history = history['valid_loss']
        train_loss_history = history['train_loss']
        valid_psnr_history = history['valid_psnr']
        train_psnr_history = history['train_psnr']
        epoch_prev = len(train_loss_history)
        print(f'CONTINUE TRAINING FROM EPOCH {epoch_prev}')
    else:
        # initialize best loss, start from epoch 0
        best_loss = np.inf
        best_epoch = 0
        epoch_prev = 0
        # initialize history
        valid_loss_history = []
        train_loss_history = []
        valid_psnr_history = []
        train_psnr_history = []
            
    # initialize best loss and weights
    best_model_wts = copy.deepcopy(model.state_dict())
    tstart = time() # start training timer

    # set up scheduler
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, last_epoch=epoch_prev-1)
    
    # loop through epochs
    for epoch in range(epoch_prev, NUM_EPOCHS):
        print('-'*50)
        print(f'TRAINING: EPOCH {epoch+1} OF {NUM_EPOCHS}')
        print('-'*50)
        epoch_start = time() # start epoch timer
            
        for phase in ['train','valid']:
            print(f'\n{phase}:')
            # set to train or eval
            if phase == 'train':
                model.train()
            else:
                model.eval()
                if ema_model is not None:
                    ema_model.eval()
            running_loss = 0.0 # initialize epoch loss
            running_psnr = 0.0 # initialize epoch psnr

            # loop through batches
            for i in range(len(dataloaders[phase])):
                # get input, move to device
                inputs, targets, prescan_norm = dataloaders[phase].__getitem__(i)
                
                # print progress
                if i % np.floor(0.2*len(dataloaders[phase])) == 0:
                    epoch_time = time() - epoch_start 
                    print(f'Sample {i+1} of {len(dataloaders[phase])} ({epoch_time:.2f}s since start of epoch)')

                # set mask with which we calculate loss
                if config['disjoint_loss']:
                    loss_mask = 1 - inputs[2]
                else:
                    loss_mask = torch.ones_like(inputs[2])
                
                optimizer.zero_grad() # zero out gradient
                
                # track history for train
                with torch.set_grad_enabled(phase=='train'):
                    if phase != 'train' and ema_model is not None:
                        yhat = ema_model(inputs)
                    else:
                        yhat = model(inputs) # make prediction

                    if config['kspace_loss'] > 0: # kspace loss
                        Ayhat = A(yhat, loss_mask, *inputs[3:])

                        while loss_mask.ndim < inputs[1].ndim:
                            loss_mask = loss_mask.unsqueeze(1)
                        Atar = inputs[1] * loss_mask
                        loss = model.criterion(Ayhat, Atar)

                        # penalize non-zero in the corners of cartesian kx ky space
                        if config['corner_penalty']:
                            yhat_k_cart = fftnc_torch(yhat, axes=[-2,-1])
                            corner_mask = torch.from_numpy(1 - circle_mask(targets.shape[-1])).to(device)
                            loss += 1e0 * mse(yhat_k_cart * corner_mask, torch.zeros_like(yhat_k_cart))
                        
                        
                    if config['kspace_loss'] != 1: # image loss
                        if ncha == 1:
                            # xhat = AH(inputs[1], loss_mask, inputs[3], traj, dcf)
                            image_loss = model.criterion(yhat, targets) # calculate loss
                        else:
                            # xhat = AH(inputs[1], inputs[2], inputs[3], loss_mask)
                            if config['TE_loss']:
                                image_loss = model.criterion(yhat, targets, inputs[3]) # calculate loss
                            else:
                                image_loss = model.criterion(yhat, targets) # calculate loss

                    if config['kspace_loss'] == 2: # combine kspace and image losses
                        loss = config['gamma'] * loss + (1-config['gamma']) * image_loss
                    elif config['kspace_loss'] == 0: # image loss only
                        loss = image_loss

                if phase == 'train':
                    loss.backward() # backprop
                    optimizer.step()
                    if ema_model is not None:
                        ema_model.update_parameters(model)

                    
                ######## QC ######
                if i == 0:
                    prefix = f'epoch{epoch}_'
                    sample = 0 #targets.shape[0]-1
                    qcdir = f'{out_dir}/qc/'
                    epoch_qc(config, qcdir, yhat, targets, inputs, loss_mask, prefix, sample, prescan_norm)
                        
                    # pdb.set_trace()
                #################

                
                running_loss += loss.item() * inputs[0].size(0) # keep track of loss
                
                # calculate psnr with gradient always disabled
                with torch.set_grad_enabled(False):
                    if isinstance(PSNR, TE_PSNR):
                        D = inputs[-1][0,]
                        running_psnr += PSNR(targets, yhat, D, 1).item() * inputs[0].size(0)
                    else:
                        running_psnr += PSNR(targets.abs(), yhat.abs(), 1).item() * inputs[0].size(0)
                
            # report loss
            epoch_loss = running_loss / dataloaders[phase].n_samples
            epoch_psnr = running_psnr / dataloaders[phase].n_samples
            print(f'{phase} loss: {epoch_loss:.4f}')
            print(f'{phase} PSNR: {epoch_psnr:.4f}')

            # store loss history
            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                valid_psnr_history.append(epoch_psnr)
                # deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_psnr = epoch_psnr
                    
                    ## save best model
                    if ema_model is not None:
                        ema_model_state = ema_model.state_dict()
                    else:
                        ema_model_state = None
                    if not os.path.exists(f'{out_dir}/models'):
                        os.makedirs(f'{out_dir}/models')
                    history = {'valid_loss':valid_loss_history,
                               'train_loss':train_loss_history,
                               'valid_psnr':valid_psnr_history,
                               'train_psnr':train_psnr_history}
                    torch.save({'model_state_dict': model.state_dict(),
                                'ema_model_state_dict': ema_model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_loss': best_loss,
                                'best_epoch': best_epoch,
                                'best_epoch_psnr': best_epoch_psnr,
                                'history':history}, f'{out_dir}/models/bestLOSS.pth')
                
                # save to tensorboard
                if tensorboard:
                    writer.add_scalar('Loss/valid', epoch_loss, epoch)
                    writer.add_scalar('PSNR/valid', epoch_psnr, epoch)
                    if inputs[0].size(1) > 1:
                        inp = inputs[0][0,0,].detach().abs()
                        out = yhat[0,0,].detach().abs()
                        tar = targets[0,0,].detach().abs()
                    else:
                        inp = inputs[0].detach().abs()
                        out = yhat.detach().abs()
                        tar = targets.detach().abs()
                    grid = torchvision.utils.make_grid(inp, normalize=True, value_range=(0,inp.max()*0.9))
                    writer.add_image('input/valid', grid, epoch)
                    grid = torchvision.utils.make_grid(out, normalize=True, value_range=(0,out.max()*0.9))
                    writer.add_image('prediction/valid', grid, epoch)
                    grid = torchvision.utils.make_grid(tar, normalize=True, value_range=(0,tar.max()*0.9))
                    writer.add_image('target/valid', grid, epoch)
                
            else:
                train_loss_history.append(epoch_loss)
                train_psnr_history.append(epoch_psnr)
                if tensorboard:
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('PSNR/train', epoch_psnr, epoch)

            # shuffle indices before next run through
            dataloaders[phase].on_epoch_end()

        # updtate learning rate with scheduler
        if use_scheduler:
            print(f'Learning rate = {scheduler.get_lr()[0]}')
            scheduler.step()
        
        # print dc layer weight
        tmp = [p for p in model.dc.parameters()]
        for ii in range(len(tmp)):
            print(f'DC parameter {ii} weight = {tmp[ii].item():.4f}')
        
        # print results from that epoch
        epoch_time = time() - epoch_start 
        print(f'\nEPOCH {epoch+1} completed in {epoch_time:.2f} seconds')
        print(f'Best loss: {best_loss:.4f} from Epoch {best_epoch+1}')
        print(f'Epoch {best_epoch+1} PSNR: {best_epoch_psnr:.4f}\n')
        # save loss curve
        if save_train_curve and epoch > 0:
            plot_loss_curve(train_loss_history, valid_loss_history, out_dir)
            plot_psnr_curve(train_psnr_history, valid_psnr_history, out_dir)
                
        ## save most recent model
        if ema_model is not None:
            ema_model_state = ema_model.state_dict()
        else:
            ema_model_state = None
        history = {'valid_loss':valid_loss_history,
                   'train_loss':train_loss_history,
                   'valid_psnr':valid_psnr_history,
                   'train_psnr':train_psnr_history}
        if not os.path.exists(f'{out_dir}/models'):
            os.makedirs(f'{out_dir}/models')
        torch.save({'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'best_epoch_psnr': best_epoch_psnr,
                    'history':history}, f'{out_dir}/models/EPOCH_{epoch:03d}.pth')

    # load best model weights
    model.load_state_dict(best_model_wts)

    print(f'Training completed in {time()-tstart:.2f} seconds\n')

    if tensorboard:
        writer.flush()
    
    return model, ema_model

