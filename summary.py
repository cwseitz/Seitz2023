import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from pycromanager import Dataset
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.io import imread
from smlm.plot import anno_blob
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte, img_as_uint, map_array
from skimage.measure import regionprops_table

class Summary:
    def __init__(self,datapath,analpath,prefix,cell_filters,nucleus_filters,z0=4):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
        self.z0 = z0
        self.cell_filters = cell_filters
        self.nucleus_filters = nucleus_filters
        dataset = Dataset(self.datapath+self.prefix)
        self.rawdata = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = self.rawdata.shape
        self.rawdata = self.rawdata.reshape((nz,nc,nt**2,nx,ny))
        self.rawdata = self.rawdata[:,:,:,:1844,:1844]
        self.ch0_sfmx = np.load(self.analpath+self.prefix+'/'+self.prefix+'_ch0_softmax.npz')['arr_0']
        self.ch1_sfmx = np.load(self.analpath+self.prefix+'/'+self.prefix+'_ch1_softmax.npz')['arr_0']
        self.ch1_spots = pd.read_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        self.ch2_spots = pd.read_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')
        print('Loaded.')
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def assign_spots_to_cells(self,ch1_spotst,ch2_spotst,ch1_mask):
        ch1_idx = np.round(ch1_spotst[['x','y']].to_numpy()).astype(np.int16)
        ch2_idx = np.round(ch2_spotst[['x','y']].to_numpy()).astype(np.int16)
        ch1_spot_labels = ch1_mask[ch1_idx[:,0],ch1_idx[:,1]]
        ch2_spot_labels = ch1_mask[ch2_idx[:,0],ch2_idx[:,1]]
        ch1_spotst = ch1_spotst.assign(cell=ch1_spot_labels)
        ch2_spotst = ch2_spotst.assign(cell=ch2_spot_labels)
        return ch1_spotst, ch2_spotst
    def plot_prob(self,output,image,mask,maskt_filtered):
        output = F.softmax(output,dim=1)
        fig,ax = plt.subplots(2,3,figsize=(8,4),sharex=True,sharey=True)
        rgb = mark_boundaries(50*image,mask,mode='thick',color=(1,1,1))
        im0 = ax[0,0].imshow(rgb,cmap='gray')
        ax[0,0].set_title('Raw')
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        im0 = ax[0,1].imshow(mask,cmap='gray')
        ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
        ax[0,1].set_title('Mask')
        im0 = ax[0,2].imshow(maskt_filtered,cmap='gray')
        ax[0,2].set_title('Filtered')
        ax[0,2].set_xticks([]); ax[0,2].set_yticks([])
        im1 = ax[1,0].imshow(output[0,0,:,:].numpy(),cmap='coolwarm')
        ax[1,0].set_title('Background')
        ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
        plt.colorbar(im1,ax=ax[1,0],label='Probability')
        im2 = ax[1,1].imshow(output[0,1,:,:].numpy(),cmap='coolwarm')
        ax[1,1].set_title('Interior')
        ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
        plt.colorbar(im2,ax=ax[1,1],label='Probability')
        im3 = ax[1,2].imshow(output[0,2,:,:].numpy(),cmap='coolwarm')
        ax[1,2].set_xticks([]); ax[1,2].set_yticks([])
        ax[1,2].set_title('Boundary')
        plt.colorbar(im3,ax=ax[1,2],label='Probability')
        plt.tight_layout()
        plt.show() 
    def get_mask(self,output):
        output = F.softmax(output,dim=1)
        nmask = np.zeros((256,256),dtype=np.bool)
        nmask[output[0,1,:,:] > 0.95] = True
        nmask = img_as_bool(nmask)
        nmask = clear_border(nmask)
        return nmask 
    def summarize(self):
        nz,nc,nt,nx,ny = self.rawdata.shape
        for n in range(nt):
            ch1_spotst = self.ch1_spots.loc[self.ch1_spots['tile'] == n]
            ch2_spotst = self.ch2_spots.loc[self.ch2_spots['tile'] == n]
            ch0_mask_filtered = self.filter_objects(self.ch0_mask[n],**self.nucleus_filters)
            ch1_mask_filtered = self.filter_objects(self.ch1_mask[n],**self.cell_filters)
            ch0_mask_filtered[ch0_mask_filtered > 0] = 1
            ch0_mask_filtered = ch0_mask_filtered*ch1_mask_filtered
            ch1_spotst = ch1_spotst.loc[ch1_spotst['x'].between(1,1843)]
            ch1_spotst = ch1_spotst.loc[ch1_spotst['y'].between(1,1843)]
            ch2_spotst = ch2_spotst.loc[ch2_spotst['x'].between(1,1843)]
            ch2_spotst = ch2_spotst.loc[ch2_spotst['y'].between(1,1843)]
            ch1_spotst, ch2_spotst = self.assign_spots_to_cells(ch1_spotst,ch2_spotst,ch1_mask_filtered)
            ch1_spotst = ch1_spotst.loc[ch1_spotst['cell'] != 0]
            ch2_spotst = ch2_spotst.loc[ch2_spotst['cell'] != 0]
            fig, ax = plt.subplots(figsize=(8,8)) 
            rgb = self.get_rgb(self.rawdata[self.z0,0,n,:,:],
                               self.rawdata[self.z0,1,n,:,:],
                               self.rawdata[self.z0,2,n,:,:])
            rgb = mark_boundaries(rgb,ch0_mask_filtered,mode='thick',color=(1,1,1))
            rgb = mark_boundaries(rgb,ch1_mask_filtered,mode='thick',color=(1,1,1))
            ax.imshow(rgb)
            anno_blob(ax,ch1_spotst,color='cyan')
            anno_blob(ax,ch2_spotst,color='yellow')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(self.analpath+self.prefix+'/'+self.prefix+'_summary.png')
            plt.show()
    
    def filter_objects(self,mask,min_area=5000,max_area=50000,min_solid=0.9):
        props = ('label', 'area', 'solidity')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask,input_labels,output_labels)
        filtered_table = regionprops_table(filtered_mask,properties=props)
        table = pd.DataFrame(table)
        filtered_table = pd.DataFrame(filtered_table)
        return filtered_mask
        
    def plot_groups(self,ax,spotdf):
        print(spotdf)
        palette = list(mcd.XKCD_COLORS.values())[::10]
        groups = spotdf.groupby(['cell'])
        for i,(name, group) in enumerate(groups):
            ax.scatter(group['y'],group['x'],marker='o',s=3,color=palette[i])
        plt.show()
