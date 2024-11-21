import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
import merlinth
from merlinth.losses.pairwise_loss import psnr
import pdb


# functional to perform complex dropout
def complex_dropout(x, p=0.5, training=True, inplace=False):
    # work around unimplemented dropout for complex
    if x.is_complex():
        mask = torch.nn.functional.dropout(torch.ones_like(x.real), p=p,
                                           training=training, inplace=inplace)
        return x * mask
    else:
        return torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)

# layer that performs complex dropout
class ComplexDropout(nn.Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return complex_dropout(input, self.p, self.training, self.inplace)

# ReLU that takes complex input and gives real output
class C2R_ReLU(torch.nn.Module):
    def forward(self, z):
        actre = torch.nn.ReLU()(torch.real(z))
        actim = torch.nn.ReLU()(torch.imag(z))
        return torch.abs(torch.complex(actre, actim))

    @property
    def __name__(self):
        return 'C2R_ReLU'

    
def mse(gt, pred, batch=True, reduce=True, hdr=False, hdr_eps=1e-3):
    """ torch mse for batch input"""
    if batch:
        batch_size = gt.shape[0]
    else:
        batch_size = 1

    # reshape the view
    pred = pred.contiguous().view(batch_size, -1)
    gt = gt.contiguous().view(batch_size, -1)
    
    if hdr: # high dynamic range loss
        error = (torch.linalg.vector_norm((gt - pred) / (pred.detach().abs() + hdr_eps), ord=2, dim=1)) ** 2
    else:
        error = (torch.linalg.vector_norm(gt - pred, ord=2, dim=1)) ** 2
    
    if reduce:
        return error.mean()
    else:
        return error
    
def mae(gt, pred, batch=True, reduce=True, hdr=False, hdr_eps=1e-3):
    """ torch mae for batch input"""
    if batch:
        batch_size = gt.shape[0]
    else:
        batch_size = 1

    # reshape the view
    pred = pred.contiguous().view(batch_size, -1)
    gt = gt.contiguous().view(batch_size, -1)

    if hdr: # high dynamic range loss
        error = torch.linalg.vector_norm((gt - pred) / (pred.detach().abs() + hdr_eps), ord=1, dim=1)
    else:
        error = torch.linalg.vector_norm(gt - pred, ord=1, dim=1)
    
    if reduce:
        return error.mean()
    else:
        return error

def ncc(gt, pred, batch=True, reduce=True):
    """ normalized cross correlation """
    if batch:
        batch_size = gt.shape[0]
    else:
        batch_size = 1

    # reshape the view
    pred = pred.contiguous().view(batch_size, -1)
    gt = gt.contiguous().view(batch_size, -1)

    error = (gt * pred).sum() / torch.numel(gt) / torch.std(gt) / torch.std(pred)
    
    if reduce:
        return error.mean()
    else:
        return error
    
class Weighted_L1L2(nn.Module):
    """
    Computes the weighted sum of L1 and L2 losses

    """
    def __init__(self, alpha=0.5, beta=0.5, batch=True, reduce=True, magnitude=False, hdr=False, hdr_eps=1e-3):
        super(Weighted_L1L2, self).__init__()
        self.batch = batch
        self.reduce = reduce
        self.alpha = alpha
        self.beta = beta
        self.magnitude = magnitude
        self.hdr = hdr
        self.hdr_eps = hdr_eps

    def forward(self, pred, gt):
        if self.magnitude:
            gt = gt.abs()
            pred = pred.abs()
        l1 = mae(gt, pred, batch=self.batch, reduce=self.reduce, hdr=self.hdr, hdr_eps=self.hdr_eps) / mae(gt, torch.zeros_like(gt))
        l2 = mse(gt, pred, batch=self.batch, reduce=self.reduce, hdr=self.hdr, hdr_eps=self.hdr_eps) / mse(gt, torch.zeros_like(gt))
        
        return self.alpha*l1 + self.beta*l2
    

    
class TE_Weighted_L1L2(nn.Module):
    """
    Computes the weighted sum of L1 and L2 losses

    """
    def __init__(self, alpha=0.5, beta=0.5, batch=True, reduce=True, magnitude=False):
        super(TE_Weighted_L1L2, self).__init__()
        self.batch = batch
        self.reduce = reduce
        self.alpha = alpha
        self.beta = beta
        self.magnitude = magnitude

    def forward(self, pred, gt, basis=None):
        if basis is not None:
            basis = basis.to(gt.device)
            gt = torch.reshape(gt, (gt.shape[0], gt.shape[1], gt.shape[2]*gt.shape[3]))
            pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3]))
            gt = torch.matmul(basis, gt)
            pred = torch.matmul(basis, pred)
        
        if self.magnitude:
            gt = gt.abs()
            pred = pred.abs()
        
        l1 = mae(gt, pred, batch=self.batch, reduce=self.reduce) / mae(gt, torch.zeros_like(gt))
        l2 = mse(gt, pred, batch=self.batch, reduce=self.reduce) / mse(gt, torch.zeros_like(gt))
        return self.alpha*l1 + self.beta*l2


class TE_PSNR(nn.Module):
    """
    Computes the weighted sum of L1 and L2 losses

    """
    def __init__(self, batch=True, reduce=True, magnitude=False):
        super(TE_PSNR, self).__init__()
        self.batch = batch
        self.reduce = reduce
        self.magnitude = magnitude

    def forward(self, gt, pred, basis, data_range=None):
        basis = basis.to(gt.device)
        gt = torch.reshape(gt, (gt.shape[0], gt.shape[1], gt.shape[2]*gt.shape[3]))
        pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3]))
        gt = torch.matmul(basis, gt)
        pred = torch.matmul(basis, pred)
        
        if self.magnitude:
            gt = gt.abs()
            pred = pred.abs()

        if data_range is None:
            # data_range = gt.abs().max()
            data_range = 1
            
        return psnr(
            gt,
            pred,
            data_range=data_range,
            batch=self.batch,
            reduce=self.reduce,
        )


    
# ReLU that takes complex input and gives real output
class C2R_ReLU(torch.nn.Module):
    def forward(self, z):
        actre = torch.nn.ReLU()(torch.real(z))
        actim = torch.nn.ReLU()(torch.imag(z))
        return torch.abs(torch.complex(actre, actim))

    @property
    def __name__(self):
        return 'C2R_ReLU'
