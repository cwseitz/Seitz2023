from pycromanager import Dataset
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Tiler:
    def __init__(self,datapath,analpath,prefix,overlap=204):
        self.overlap = overlap
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
    def tile(self):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath + self.prefix + '_mxtiled_ch0.tif',ch0)
        del ch0
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath + self.prefix + '_mxtiled_ch1.tif',ch1)
        del ch1
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath + self.prefix + '_mxtiled_ch2.tif',ch2)
        del ch2

