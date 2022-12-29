import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
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
    def __init__(self,config,model_path,stack_path,prefix,z0=5):
        self.config = config
        self.stack_path = stack_path
        self.prefix = prefix
        self.z0 = z0
        self.model_path = model_path

	def softmax_and_resize(output):
		output = F.softmax(output,dim=1)
		idx = output.argmax(dim=1)
		mask = F.one_hot(idx,num_classes=3)
		mask = torch.permute(mask,(0,3,1,2))
		nmask = mask[0,1,:,:].numpy()
		nmask = clear_border(nmask)
		nmask_full = resize(nmask,(2048,2048))
		nmask_full[nmask_full > 0] = 1
		nmask_full = label(nmask_full)
		return nmask_full

	def preprocess(ch0):
		ch0 = resize(ch0,(256,256),preserve_range=True)
		background = rolling_ball(ch0)
		ch0 = ch0 - background
		ch0 = gaussian(ch0, sigma=0.5, preserve_range=True)
		return ch0

	def segment(config,model_path,stack_path,prefix,z0=5):
		dataset = Dataset(stack_path)
		X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
		nz,nc,nt,_,nx,ny = X.shape
		X = X.reshape((nz,nc,nt**2,nx,ny))
		model_path = model_path + 'model_best.pth'
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = config.init_obj('arch', module_arch)
		model.to(device=device)
		checkpoint = torch.load(model_path, map_location=device)
		model.load_state_dict(checkpoint['state_dict'])
		model.eval()
		mask = np.zeros((nt**2,nx,ny),dtype=np.int16)
		for n in range(nt**2):
		    with torch.no_grad():
		        print(f'Segmenting tile {n}')
		        x = preprocess(X[z0,0,n,:,:])
		        image = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
		        image = image.to(device=device, dtype=torch.float)
		        output = model(image).cpu()
		        mask[n] = softmax_and_resize(output)
		        torch.cuda.empty_cache()
		stack_path = stack_path.replace('Data','Analysis')
		imsave(stack_path + prefix + '_ch0_mask.tif', mask)

