from smlm.localize import LOGDetector
from smlm.filters import blur
from smlm.plot import anno_blob
from pycromanager import Dataset
from skimage.filters import gaussian, median
from skimage.measure import regionprops_table
from skimage.util import map_array, img_as_int
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import napari
import tifffile
import warnings
warnings.filterwarnings('ignore')

class Detector:
    def __init__(self,datapath,analpath,prefix,ch1_thresh,ch2_thresh):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
        self.ch1_thresh = ch1_thresh
        self.ch2_thresh = ch2_thresh
    def detect(self,plot=False,randomize=True):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        zrange = np.arange(3,6)
        X = X.reshape((nz,nc,nt**2,nx,ny))
        X = X[:,:,:,:1844,:1844]
        ch1_blobs = pd.DataFrame()
        ch2_blobs = pd.DataFrame()
        narray = np.arange(0,nt**2)
        if randomize:
            np.random.shuffle(narray)
        for n in narray:
            print(f'Detecting in tile {n}')
            ch1 = np.max(np.array(X[:,1,n,:,:]),axis=0)
            ch1det = LOGDetector(ch1,threshold=self.ch1_thresh)
            ch1_blobst = ch1det.detect()
            ch1_blobst = ch1_blobst.assign(tile=n)
            ch1_blobs = pd.concat([ch1_blobs,ch1_blobst])
            ch2 = np.max(np.array(X[:,2,n,:,:]),axis=0)
            ch2_filt = median(ch2)
            ch2_filt= img_as_int(ch2_filt)
            ch2det = LOGDetector(ch2_filt,threshold=self.ch2_thresh)
            ch2_blobst = ch2det.detect()
            if plot:
                fig, ax = plt.subplots()
                ax.imshow(10*ch2,cmap='gray')
                anno_blob(ax,ch2_blobst,color='red')
                plt.show()
            ch2_blobst = ch2_blobst.assign(tile=n)
            ch2_blobs = pd.concat([ch2_blobs,ch2_blobst])
        ch1_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        ch2_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')



