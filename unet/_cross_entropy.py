import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage.measure import label
from skimage.io import imread
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries
from skimage.color import rgb2gray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def unet_weight_map(y, wc=None, w0=10, sigma=5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.
    
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    
    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)
    
    return w

def CrossEntropyLoss(output, target, diagnostic=True):

    if diagnostic: 
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(output[0,0,:,:].detach().cpu().numpy())
        ax[1].imshow(target[0,0,:,:].detach().cpu().numpy())
        plt.show()
    return F.cross_entropy(output,target)

def WeightedCrossEntropyLoss(output, target, weighted=False, diagnostic=False):
    """
    Reduces output to a single loss value by averaging losses at each element
    by default in (PyTorch 1.11.0)
    
    ***Note: this function operates on non-normalized probabilities
    (there is no need for a softmax function, it is included in the loss).
    
    At testing time or when computing metrics, you will need to implement a softmax layer.
    """
    
    wc = {
    0: 0, # background
    1: 0  # objects
    }
    
    mat = target.cpu().numpy()[:,1,:,:]
    
    if weighted:
        weight = torch.zeros_like(target[:,1,:,:])
        for idx in range(mat.shape[0]):
    	    weight[idx] = torch.from_numpy(unet_weight_map(mat[idx], wc))
    else:
        weight = torch.ones_like(target[:,1,:,:])

    logp = -torch.log(torch.sum(torch.mul(F.softmax(output,dim=1),target),dim=1))
    if torch.cuda.is_available():
    	weight = weight.cuda()

    loss_map = torch.mul(logp,weight)

    if diagnostic:
        im1 = torch.mul(F.softmax(output,dim=1),target)[0].permute(1,2,0).cpu().detach().numpy()
        im2 = target[0].permute(1,2,0).cpu().detach().numpy()
        im3 = loss_map[0].cpu().detach().numpy()
        fig, ax = plt.subplots(1,3,sharex=True,sharey=True)
        ax[0].imshow(im2,cmap='gray')
        ax[0].set_title('target')
        ax[1].imshow(im1,cmap='gray')
        ax[1].set_title('output')
        ax[2].imshow(10*im3,cmap='coolwarm')
        ax[2].set_title('weights')
        for x in ax:
            x.axis('off')
        plt.show()
    loss = torch.mean(loss_map)

    return loss
