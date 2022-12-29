from pycromanager import Dataset
from basicpy import BaSiC
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Basic:
    def __init__(self,analpath,prefix):
        self.analpath = analpath
        self.prefix = prefix
    def blockshaped(self,arr,nrows,ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def correct(self):
        nt,nx,ny = 10,1844,1844
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch0.tif'
        ch0 = tifffile.imread(path)
        ch0_blocks = self.blockshaped(ch0,1844,1844)
        basic = BaSiC(get_darkfield=True)
        basic.fit(ch0_blocks)
        ch0_correct = basic.transform(ch0_blocks)
        ch0_correct = ch0_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch0.tif'
        tifffile.imwrite(path,ch0_correct)
        ch0_correct = ch0_correct.reshape((nt,nt,nx,ny))
        ch0_correct = ch0_correct.swapaxes(1,2)
        ch0_correct = ch0_correct.reshape((nt*nx,nt*ny))
        ch0_correct = ch0_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_ch0.tif'
        tifffile.imwrite(path,ch0_correct)

        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch1.tif'
        ch1 = tifffile.imread(path)
        ch1_blocks = self.blockshaped(ch1,1844,1844)
        basic = BaSiC(get_darkfield=True)
        basic.fit(ch1_blocks)
        ch1_correct = basic.transform(ch1_blocks)
        ch1_correct = ch1_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch1.tif'
        tifffile.imwrite(path,ch1_correct)
        ch1_correct = ch1_correct.reshape((nt,nt,nx,ny))
        ch1_correct = ch1_correct.swapaxes(1,2)
        ch1_correct = ch1_correct.reshape((nt*nx,nt*ny))
        ch1_correct = ch1_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_ch1.tif'
        tifffile.imwrite(path,ch1_correct)

        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch2.tif'
        ch2 = tifffile.imread(path)
        ch2_blocks = self.blockshaped(ch2,1844,1844)
        basic = BaSiC(get_darkfield=True)
        basic.fit(ch2_blocks)
        ch2_correct = basic.transform(ch2_blocks)
        ch2_correct = ch2_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_stack_ch2.tif'
        tifffile.imwrite(path,ch2_correct)
        ch2_correct = ch2_correct.reshape((nt,nt,nx,ny))
        ch2_correct = ch2_correct.swapaxes(1,2)
        ch2_correct = ch2_correct.reshape((nt*nx,nt*ny))
        ch2_correct = ch2_correct.astype(np.uint16)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_ch2.tif'
