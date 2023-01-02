from smlm.localize import LOGDetector
from smlm.filters import blur
from pycromanager import Dataset
from skimage.filters import gaussian
from skimage.measure import regionprops_table
from skimage.util import map_array
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
    def detect(self,ch1_thres=0.003,ch2_thres=0.00175,plot=False):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        zrange = np.arange(3,6)
        X = X.reshape((nz,nc,nt**2,nx,ny))
        X = X[:,:,:,:1844,:1844]
        ch1_blobs = pd.DataFrame()
        ch2_blobs = pd.DataFrame()
        for n in range(nt**2):
            #viewer = napari.view_image(X[:,1,n,:,:], colormap='magma')
            for z in zrange:
                print(f'Detecting in tile {n}, plane {z}')
                ch1 = gaussian(np.array(X[z,1,n,:,:]),sigma=1)
                ch1det = LOGDetector(ch1,threshold=self.ch1_thresh)
                ch1_blobst = ch1det.detect()
                if plot:
                    ch1det.show()
                ch1_blobst = ch1_blobst.assign(tile=n,z=z)
                ch1_blobs = pd.concat([ch1_blobs,ch1_blobst])
                ch2 = gaussian(np.array(X[z,2,n,:,:]),sigma=1)
                self.ch2_thresh = self.ch2_thresh*(ch2.mean()/0.0016)
                ch2det = LOGDetector(ch2,threshold=self.ch2_thresh)
                ch2_blobst = ch2det.detect()
                if plot:
                    ch2det.show()
                ch2_blobst = ch2_blobst.assign(tile=n,z=z)
                ch2_blobs = pd.concat([ch2_blobs,ch2_blobst])
        ch1_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        ch2_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')



