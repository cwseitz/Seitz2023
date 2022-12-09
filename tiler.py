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
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        #np.savez(self.opath + self.prefix + '_mxtiled_ch0.npz',ch0=ch0)
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch0.tif',ch0)
        del ch0
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        #np.savez(self.opath + self.prefix + '_mxtiled_ch1.npz',ch1=ch1)
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch1.tif',ch1)
        del ch1
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        #np.savez(self.opath + self.prefix + '_mxtiled_ch2.npz',ch2=ch2)
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch2.tif',ch2)
        del ch2
        # print('Tiling Channel 3\n')
        # ch3 = np.asarray(X[:,3,:,:,:-self.overlap,:-self.overlap])
        # ch3 = np.max(ch3,axis=0) #max intensity projection
        # np.savez(self.opath + self.prefix + '_mxtiled_ch3.npz',ch3=ch3)
        # ch3 = ch3.swapaxes(1,2)
        # ch3 = ch3.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        # tifffile.imwrite(self.opath + self.prefix + '_mxtiled_ch3.tif',ch3)
        # del ch3
