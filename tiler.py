from pycromanager import Dataset
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Tiler:
    def __init__(self,ipath,opath,prefix,overlap=204):
        self.overlap = overlap
        self.ipath = ipath
        self.opath = opath
        self.prefix = prefix
    def tile(self):
        dataset = Dataset(self.ipath)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        #X = X.reshape((nz,nc,nt**2,nx,ny))
        #stack = 1*X[:,0,:10,:,:] + 0*X[:,1,:10,:,:]
        #tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch0_test.tif',stack,imagej=True)
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch0.tif',ch0)
        del ch0
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch1.tif',ch1)
        del ch1
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch2.tif',ch2)
        del ch2

