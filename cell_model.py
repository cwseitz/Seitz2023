import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
import UNET.data_loaders as data_loaders
import pandas as pd
from pycromanager import Dataset
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.torch_models import UNetModel
from torchsummary import summary
from skimage.io import imread, imsave
from skimage.segmentation import clear_border, mark_boundaries
from skimage.measure import label
from skimage.util import map_array, img_as_int, img_as_bool
from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.measure import regionprops_table

class CellBodyModel:
    def __init__(self,cmodelpath,analpath,prefix,cfilters):
        self.analpath = analpath
        self.prefix = prefix
        self.cmodelpath = cmodelpath
        self.cfilters = cfilters
    def plot_prob(self,output,image,mask,maskt_filtered):
        output = F.softmax(output,dim=1)
        fig,ax = plt.subplots(2,3,figsize=(8,4),sharex=True,sharey=True)
        rgb = mark_boundaries(50*image,mask,mode='thick',color=(1,1,1))
        im0 = ax[0,0].imshow(rgb,cmap='gray')
        ax[0,0].set_title('Raw')
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        im0 = ax[0,1].imshow(mask,cmap='gray')
        ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
        ax[0,1].set_title('Mask')
        im0 = ax[0,2].imshow(maskt_filtered,cmap='gray')
        ax[0,2].set_title('Filtered')
        ax[0,2].set_xticks([]); ax[0,2].set_yticks([])
        im1 = ax[1,0].imshow(output[0,0,:,:].numpy(),cmap='coolwarm')
        ax[1,0].set_title('Background')
        ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
        plt.colorbar(im1,ax=ax[1,0],label='Probability')
        im2 = ax[1,1].imshow(output[0,1,:,:].numpy(),cmap='coolwarm')
        ax[1,1].set_title('Interior')
        ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
        plt.colorbar(im2,ax=ax[1,1],label='Probability')
        im3 = ax[1,2].imshow(output[0,2,:,:].numpy(),cmap='coolwarm')
        ax[1,2].set_xticks([]); ax[1,2].set_yticks([])
        ax[1,2].set_title('Boundary')
        plt.colorbar(im3,ax=ax[1,2],label='Probability')
        plt.tight_layout()
        plt.show()    
    def resize_and_blend(self,ch0,ch1,new_size=(256,256)):
        ch0 = resize(ch0,new_size)
        ch1 = resize(ch1,new_size)
        r = ch0.max()/ch1.max()
        ch1 *= r
        blended = 0.4*ch0 + 0.6*ch1
        blended = gaussian(blended,sigma=1)
        blended = rolling_ball(blended,radius=50)
        blended = img_as_int(2*blended)
        return blended
    def apply(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetModel(1,3)
        model.to(device=device)
        checkpoint = torch.load(self.cmodelpath+'model_best.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch0.tif'
        ch0 = tifffile.imread(path)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch1.tif'
        ch1 = tifffile.imread(path)
        nt,nx,ny = ch0.shape
        prob = np.zeros((nt,3,256,256))
        for n in range(nt):
            with torch.no_grad():
                print(f'Segmenting tile {n}')
                x = self.resize_and_blend(ch0[n],ch1[n])
                image = (x-x.mean())/x.std()
                image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
                image = image.to(device=device, dtype=torch.float)
                output = model(image).cpu()
                prob[n] = F.softmax(output,dim=1)
                torch.cuda.empty_cache()
        np.savez(self.analpath+self.prefix+'/'+self.prefix+'_ch1_softmax.npz',prob)

