import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
import UNET.data_loaders as data_loaders
from pycromanager import Dataset
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.torch_models import UNetModel
from torchsummary import summary
from skimage.io import imread, imsave
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
torch.cuda.empty_cache()

class NucleusModel:
    def __init__(self,nmodelpath,analpath,prefix):
        self.analpath = analpath
        self.prefix = prefix
        self.nmodelpath = nmodelpath
    def softmax_and_resize(self,output):
        output = F.softmax(output,dim=1)
        idx = output.argmax(dim=1)
        mask = F.one_hot(idx,num_classes=3)
        mask = torch.permute(mask,(0,3,1,2))
        nmask = mask[0,1,:,:].numpy()
        nmask = clear_border(nmask)
        nmask_full = resize(nmask,(1844,1844))
        nmask_full[nmask_full > 0] = 1
        nmask_full = label(nmask_full)
        return nmask_full
    def preprocess(self,ch0):
        ch0 = resize(ch0,(256,256),preserve_range=True)
        background = rolling_ball(ch0)
        ch0 = ch0 - background
        ch0 = gaussian(ch0,sigma=0.5,preserve_range=True)
        return ch0
    def segment(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetModel(1,3)
        model.to(device=device)
        checkpoint = torch.load(self.nmodelpath+'model_best.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch0.tif'
        ch0 = tifffile.imread(path)
        nt,nx,ny = ch0.shape
        mask = np.zeros((nt,nx,ny),dtype=np.int16)
        for n in range(nt):
            with torch.no_grad():
                print(f'Segmenting tile {n}')
                x = self.preprocess(ch0[n])
                image = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
                image = image.to(device=device, dtype=torch.float)
                output = model(image).cpu()
                mask[n] = self.softmax_and_resize(output)
                torch.cuda.empty_cache()
        imsave(self.analpath+self.prefix+'/'+self.prefix+'_ch0_mask.tif', mask)

