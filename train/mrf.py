import matplotlib.pyplot as plt
import numpy as np
import matplotlib._color_data as mcd
import tifffile
from skimage.transform import resize
from pycromanager import Dataset
from glob import glob
from skimage.io import imread
from skimage.filters import gaussian

import numpy as np
import pandas as pd
import os, os.path
from scipy import misc
import sys
from matplotlib.pyplot import imshow
import imageio
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import optimize
import random
import warnings
warnings.filterwarnings('ignore')

class MRFSegmenter:
    def __init__(self,ipath,opath,prefix):
        self.ipath = ipath
        self.opath = opath
        self.mask = tifffile.imread(opath + prefix + '_ch0_mask.tif')
        self.dataset = Dataset(self.ipath)
    def segment(self,z0=4):
        X = self.dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        X = X.reshape((nz,nc,nt**2,nx,ny))
        image = X[z0,1,0,:,:]
        


