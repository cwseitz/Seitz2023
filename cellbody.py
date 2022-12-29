import argparse
import collections
import torch
import numpy as np
import tifffile
import torch.nn.functional as F
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
import UNET.data_loaders as data_loaders
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.torch_models import UNetModel
from torchsummary import summary
from pycromanager import Dataset
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.util import img_as_int
from skimage.restoration import rolling_ball
from skimage.io import imsave, imread

class CellBodyModel:
    def __init__(self,cmodelpath,analpath,prefix):
        self.analpath = analpath
        self.prefix = prefix
        self.cmodelpath = cmodelpath
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
    def segment(self):
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
        mask = np.zeros((nt,nx,ny),dtype=np.int16)
        for n in range(nt):
            with torch.no_grad():
                print(f'Segmenting tile {n}')
                image = self.resize_and_blend(ch0[n],ch1[n])
                image = (image-image.mean())/image.std()
                image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
                image = image.to(device=device, dtype=torch.float)
                output = model(image).cpu()
                mask[n] = self.softmax_and_resize(output)
                torch.cuda.empty_cache()
        imsave(self.analpath+self.prefix+'/'+self.prefix+'_ch1_mask.tif', mask)

