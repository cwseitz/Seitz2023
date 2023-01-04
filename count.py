import numpy as np
import pandas as pd
import tifffile
import napari
import uuid
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
    def intensity_hist(self):
        fig, ax = plt.subplots(1,2)
        ax[0].hist(self.ch1_spots['peak'],bins=200,color='red')
        ax[1].hist(self.ch2_spots['peak'],bins=10,color='blue')
        plt.show()
    def assign_spots_to_cells(self,ch1_spots,ch2_spots,mask):
        ch1_idx = np.round(ch1_spots[['x','y','tile']].to_numpy()).astype(np.int16)
        ch2_idx = np.round(ch2_spots[['x','y','tile']].to_numpy()).astype(np.int16)
        ch1_spot_labels = mask[ch1_idx[:,2],ch1_idx[:,0],ch1_idx[:,1]]
        ch2_spot_labels = mask[ch2_idx[:,2],ch2_idx[:,0],ch2_idx[:,1]]
        ch1_spots = ch1_spots.assign(label=ch1_spot_labels)
        ch1_spots = ch1_spots.loc[ch1_spots['label'] != 0]
        ch2_spots = ch2_spots.assign(label=ch2_spot_labels)
        ch2_spots = ch2_spots.loc[ch2_spots['label'] != 0]
        return ch1_spots, ch2_spots
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def filter_spots(self,ch1_spots,ch2_spots,z=None):
        ch1_avg = ch1_spots['peak'].mean()
        ch2_avg = ch2_spots['peak'].mean()
        ch1_spots = ch1_spots.loc[ch1_spots['peak'] > 0.003]
        ch2_spots = ch2_spots.loc[ch2_spots['peak'] > 0.0015]
        if z is not None:
            ch1_spots = ch1_spots.loc[ch1_spots['z'] == z]
            #ch2_spots = ch2_spots.loc[ch2_spots['z'] == z]
        return ch1_spots, ch2_spots
    def plot_spots(self,ch1_spots,ch2_spots):
        ch1_spots, ch2_spots = self.filter_spots(ch1_spots,ch2_spots)
        nz,nc,nt,nx,ny = self.rawdata.shape
        zs = ch1_spots['z'].unique()
        for n in range(nt):
            fig, ax = plt.subplots(2,len(zs),figsize=(12,6),sharex=True,sharey=True) 
            ch1_spotst = ch1_spots.loc[ch1_spots['tile'] == n]
            ch2_spotst = ch2_spots.loc[ch2_spots['tile'] == n]
            for m,z in enumerate(zs):
                ch1_spotstz =  ch1_spotst.loc[ch1_spotst['z'] == z]
                ch2_spotstz =  ch2_spotst.loc[ch2_spotst['z'] == z]
                ch1 = gaussian(self.rawdata[z,1,n,:,:],sigma=1)
                ch2 = gaussian(self.rawdata[z,2,n,:,:],sigma=1)
                ax[0,m].imshow(ch1,cmap='gray'); ax[1,m].imshow(ch2,cmap='gray')
                anno_blob(ax[0,m],ch1_spotstz,color='cyan')
                anno_blob(ax[1,m],ch2_spotstz,color='yellow')
                ax[0,m].set_xticks([]); ax[0,m].set_yticks([])
                ax[1,m].set_xticks([]); ax[1,m].set_yticks([])
            plt.tight_layout()
            plt.show()
    def map_to_uuid(self,ngroups):
        uuids = np.array([str(uuid.uuid4()) for _ in range(ngroups)])
        return uuids
    def count_matrix(self,plot=False,z=None):
        self.ch1_spots, self.ch2_spots = self.count()
        self.ch1_spots, self.ch2_spots = self.filter_spots(self.ch1_spots,self.ch2_spots,z=z)
        if plot:
            self.plot_spots(self.ch1_spots,self.ch2_spots)
        self.ch1_spots = self.ch1_spots.assign(gene='gapdh')
        self.ch2_spots = self.ch2_spots.assign(gene='gbp5')
        spots = pd.concat([self.ch1_spots,self.ch2_spots])
        grouped = spots.groupby(['tile','label'])
        ngrps = grouped.ngroup().nunique()
        uuids = self.map_to_uuid(ngrps)
        spots['cell_id'] = uuids[grouped.ngroup()]
        count_mat = pd.DataFrame([])
        for i,(name, group) in enumerate(grouped):
            ngpdh = len(group.loc[group['gene'] == 'gapdh'])
            ngbp5 = len(group.loc[group['gene'] == 'gbp5'])
            cell_id = group['cell_id'].unique()[0]
            dict = {'gapdh': ngpdh, 'gbp5': ngbp5, 'cell_id': cell_id}
            count_mat = count_mat.append(dict, ignore_index=True)
        count_mat = count_mat.assign(grid=self.prefix)
        spots = spots.assign(grid=self.prefix)
        return spots, count_mat           
    def count(self):
        print(f'Counting spots in {self.prefix}')
        nz,nc,nt,nx,ny = self.rawdata.shape
        self.ch1_spots, self.ch2_spots =\
        self.assign_spots_to_cells(self.ch1_spots,self.ch2_spots,self.ch1_mask)
        return self.ch1_spots, self.ch2_spots

    
