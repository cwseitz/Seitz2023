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

class CellSegmenter:
    def __init__(self,datapath,analpath,prefix,cell_filters,p0=0.95):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
        self.p0 = p0
        self.cell_filters = cell_filters
        dataset = Dataset(self.datapath+self.prefix)
        self.rawdata = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = self.rawdata.shape
        self.rawdata = self.rawdata.reshape((nz,nc,nt**2,nx,ny))
        self.rawdata = self.rawdata[:,:,:,:1844,:1844]
        self.ch1_sfmx = np.load(self.analpath+self.prefix+'/'+self.prefix+'_ch1_softmax.npz')['arr_0']
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = 1000*ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def mask_plot(self,sfmx,mask1,mask2,mask3):
        fig,ax = plt.subplots(2,3,figsize=(8,4))
        im0 = ax[0,0].imshow(mask1,cmap='gray')
        ax[0,0].set_title('Raw')
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        im0 = ax[0,1].imshow(mask2,cmap='gray')
        ax[0,1].set_xticks([]); ax[0,1].set_yticks([])
        ax[0,1].set_title('Mask')
        im0 = ax[0,2].imshow(mask3,cmap='gray')
        ax[0,2].set_title('Filtered')
        ax[0,2].set_xticks([]); ax[0,2].set_yticks([])
        im1 = ax[1,0].imshow(sfmx[0,:,:],cmap='coolwarm')
        ax[1,0].set_title('Background')
        ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
        plt.colorbar(im1,ax=ax[1,0],label='Probability')
        im2 = ax[1,1].imshow(sfmx[1,:,:],cmap='coolwarm')
        ax[1,1].set_title('Interior')
        ax[1,1].set_xticks([]); ax[1,1].set_yticks([])
        plt.colorbar(im2,ax=ax[1,1],label='Probability')
        im3 = ax[1,2].imshow(sfmx[2,:,:],cmap='coolwarm')
        ax[1,2].set_xticks([]); ax[1,2].set_yticks([])
        ax[1,2].set_title('Boundary')
        plt.colorbar(im3,ax=ax[1,2],label='Probability')
        plt.tight_layout()
    def get_mask(self,sfmx):
        nmask = np.zeros((256,256),dtype=np.bool)
        nmask[sfmx[1,:,:] >= self.p0] = True
        nmask = img_as_bool(nmask)
        nmask = gaussian(nmask,sigma=0.2)
        nmask[nmask > 0] = 1
        nmask = img_as_bool(nmask)
        return nmask 
    def segment(self,correct=False):
        nz,nc,nt,nx,ny = self.rawdata.shape
        label_stack = np.zeros((nt,nx,ny),dtype=np.int16)
        for n in range(nt):
            print(f'Segmenting tile {n}')
            ch1 = resize(np.max(self.rawdata[:,1,n,:,:],axis=0),(256,256))
            ch1_mask = self.get_mask(self.ch1_sfmx[n])
            ch1_mask = img_as_bool(resize(ch1_mask,(1844,1844)))
            ch1_mask_filtered = self.filter_objects(ch1_mask,**self.cell_filters)
            if correct:
                viewer = napari.Viewer()
                viewer.window.resize(1500,1000)
                mx = np.max(self.rawdata[:,1,n,:,:],axis=0)
                viewer.add_image(mx,name='GAPDH',colormap='green',visible=True)
                viewer.add_image(ch1_mask,name='Prefilter',colormap='gray',visible=True,opacity=0.1)
                contours = find_contours(ch1_mask_filtered)
                contours = [contour[::10] for contour in contours]
                viewer.add_shapes(contours, shape_type='polygon', edge_width=5,
                              edge_color='red', face_color='white', opacity=0.3, name='UNET')
                napari.run()
                labels = viewer.layers['UNET'].to_labels(labels_shape=ch1_mask.shape)
                labels = label(labels).astype(np.uint16)
            else:
                labels = label(ch1_mask_filtered).astype(np.uint16)
            label_stack[n] = labels
        imsave(self.analpath+self.prefix+'/'+self.prefix+f'_ch1_mask.tif',label_stack)
    def filter_objects(self,mask,min_area=None,max_area=None,min_solid=None):
        mask = label(mask)
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
        filtered_mask = clear_border(filtered_mask)
        filtered_mask[filtered_mask > 0]  = 1
        return filtered_mask
    def plot_groups(self,ax,spotdf):
        palette = list(mcd.XKCD_COLORS.values())[::10]
        groups = spotdf.groupby(['cell'])
        for i,(name, group) in enumerate(groups):
            ax.scatter(group['y'],group['x'],marker='o',s=3,color=palette[i])
        plt.show()
        
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
    def count(self,plot=False):
        nz,nc,nt,nx,ny = self.rawdata.shape
        self.ch1_spots, self.ch2_spots =\
        self.assign_spots_to_cells(self.ch1_spots,self.ch2_spots,self.ch1_mask)
        self.ch1_spots.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        self.ch2_spots.to_csv(self.analpath+self.prefix+'/'+self.prefix+'_ch2_spots.csv')
        if plot:
            fig, ax = plt.subplots(1,2,figsize=(12,6)) 
            ch1 = np.max(self.rawdata[:,1,n,:,:],axis=0)
            ch2 = np.max(self.rawdata[:,2,n,:,:],axis=0)
            ch1 = mark_boundaries(ch1,mask,mode='thick',color=(1,1,1))
            ch2 = mark_boundaries(ch2,mask,mode='thick',color=(1,1,1))
            ax[0].imshow(50*ch1)
            ax[1].imshow(100*ch2)
            anno_blob(ax[0],ch1_spotst,color='cyan')
            anno_blob(ax[1],ch2_spotst,color='yellow')
            ax[0].set_xticks([]); ax[0].set_yticks([])
            ax[1].set_xticks([]); ax[1].set_yticks([])
            plt.tight_layout()
            plt.show()
    
