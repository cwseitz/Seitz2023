import numpy as np
import pandas as pd
import tifffile
import napari
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from pycromanager import Dataset
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.io import imread, imsave
from smlm.plot import anno_blob
from skimage.segmentation import mark_boundaries, clear_border
from skimage.transform import resize
from skimage.util import img_as_ubyte, img_as_uint, map_array, img_as_bool
from skimage.measure import regionprops_table, label, find_contours
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

class SpotCounts:
    def __init__(self,datapath,analpath,prefix,z0=4):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
        dataset = Dataset(self.datapath+self.prefix)
        self.rawdata = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = self.rawdata.shape
        self.rawdata = self.rawdata.reshape((nz,nc,nt**2,nx,ny))
        self.rawdata = self.rawdata[:,:,:,:1844,:1844]
        self.ch1_spots = pd.read_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        self.ch2_spots = pd.read_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')
        self.ch1_mask = imread(self.analpath+self.prefix+'/'+self.prefix+'_ch1_mask.tif')
    def assign_spots_to_cells(self,ch1_spots,ch2_spots,mask):
        ch1_idx = np.round(ch1_spots[['x','y','tile']].to_numpy()).astype(np.int16)
        ch2_idx = np.round(ch2_spots[['x','y','tile']].to_numpy()).astype(np.int16)
        ch1_spot_labels = mask[ch1_idx[:,2],ch1_idx[:,0],ch1_idx[:,1]]
        ch2_spot_labels = mask[ch2_idx[:,2],ch2_idx[:,0],ch2_idx[:,1]]
        ch1_spots = ch1_spots.assign(cell=ch1_spot_labels)
        ch2_spots = ch2_spots.assign(cell=ch2_spot_labels)
        return ch1_spots, ch2_spots
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def count_matrix(self):
        self.ch1_spots, self.ch2_spots = self.count()
        self.ch1_spots = self.ch1_spots.assign(gene='gapdh')
        self.ch2_spots = self.ch2_spots.assign(gene='gbp5')
        spots = pd.concat([self.ch1_spots,self.ch2_spots])
        grouped = spots.groupby(['tile','cell'])
        ncells = grouped.ngroups
        count_mat = np.zeros((ncells,2))
        for i,(name, group) in enumerate(grouped):
            count_mat[i,0] = len(group.loc[group['gene'] == 'gapdh'])
            count_mat[i,1] = len(group.loc[group['gene'] == 'gbp5'])
        return count_mat            
    def count(self,plot=False):
        print(f'Counting spots in {self.prefix}')
        nz,nc,nt,nx,ny = self.rawdata.shape
        self.ch1_spots, self.ch2_spots =\
        self.assign_spots_to_cells(self.ch1_spots,self.ch2_spots,self.ch1_mask)
        return self.ch1_spots, self.ch2_spots

    
