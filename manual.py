import matplotlib.pyplot as plt
import numpy as np
import tifffile
import math
from smlm.track import LOGDetector
from smlm.filters import blur
from smlm.plot import anno_blob
import napari 

class Analyzer:
    def __init__(self,pfx,opath,sfx='_mxtiled_corrected_stack_'):
        self.pfx = pfx
        self.opath = opath
        self.sfx = sfx
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def plot_raw(self,ch0,ch1,ch2):
        ch1 = blur(ch1,sigma=1)
        ch2 = blur(ch2,sigma=1)
        fig, ax = plt.subplots(figsize=(10,10))
        rgb = self.get_rgb(ch0,ch1,ch2)
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    def plot_result(self,ch0,ch1,ch2,mask,ch0_df,ch1_df,ch2_df,gain=50):
        fig, ax = plt.subplots(figsize=(10,10))
        ch1 = blur(ch1,sigma=1)
        ch2 = blur(ch2,sigma=1)      
        rgb = self.get_rgb(ch0,ch1,ch2)
        rgb = mark_boundaries(rgb,mask,color=(0.8,0.8,0.8))
        ax.imshow(rgb)
        anno_blob(ax,ch0_df,marker='^',markersize=5,plot_r=False,color='blue')
        anno_blob(ax,ch1_df,marker='^',markersize=5,plot_r=False,color='green')
        anno_blob(ax,ch2_df,marker='^',markersize=20,plot_r=False,color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    def analyze(self,filters,ch1_dft=0.0005,ch2_dft=0.0005):
        ch0_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch0.tif')
        ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
        ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')
        nt,nx,ny = ch0_stack.shape
        for n in range(nt):
            viewer = napari.Viewer()
            viewer.window.resize(2000, 1000)
            viewer.add_image(ch0_stack[n],name='DAPI',colormap='blue',visible=False)
            viewer.add_image(blur(ch1_stack[n],sigma=1),name='GAPDH',colormap='green')
            viewer.add_image(blur(ch2_stack[n],sigma=1),name='GBP5',colormap='red',visible=False)
            napari.run()
            shape = ch1_stack[n].shape
            try:
                labels = viewer.layers['Shapes'].to_labels(labels_shape=shape)
                ch1_pts = viewer.layers['GAPDH-features-blob_log'].data
                ch2_pts = viewer.layers['GBP5-features-blob_log'].data
            except:
                pass

        
    def detect_spots(self,ch1,ch2,ch1_thr,ch2_thr):
        det1 = LOGDetector(blur(ch1,sigma=1),threshold=ch1_thr)
        det1.detect()
        det2 = LOGDetector(blur(ch2,sigma=1),threshold=ch2_thr)
        det2.detect()
        pts1 = det1.blobs_df[['x','y']].to_numpy()
        pts2 = det2.blobs_df[['x','y']].to_numpy()
        return pts1, pts2
        

