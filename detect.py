from smlm.localize import LOGDetector
from smlm.filters import blur
from pycromanager import Dataset
from skimage.measure import regionprops_table
from skimage.util import map_array
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Detector:
    def __init__(self,datapath,analpath,prefix):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
    def detect(self,ch1_thres=0.0003,ch2_thres=0.000175,z0=4):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        X = X.reshape((nz,nc,nt**2,nx,ny))
        ch1_blobs = pd.DataFrame()
        ch2_blobs = pd.DataFrame()
        for n in range(nt**2):
            print(f'Detecting in tile {n}')
            ch1det = LOGDetector(np.array(X[z0,1,n,:,:]),threshold=ch1_thres)
            ch1_blobst = ch1det.detect()
            ch1_blobst = ch1_blobst.assign(tile=n)
            print(ch1_blobst)
            ch1_blobs = pd.concat([ch1_blobs,ch1_blobst])
            ch2det = LOGDetector(np.array(X[z0,2,n,:,:]),threshold=ch2_thres)
            ch2_blobst = ch2det.detect()
            ch2_blobst = ch2_blobst.assign(tile=n)
            ch2_blobs = pd.concat([ch2_blobs,ch2_blobst])
        ch1_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        ch2_blobs.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')



