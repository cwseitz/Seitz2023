import matplotlib.pyplot as plt
import numpy as np
import tifffile
import math
import pandas as pd
from skimage.segmentation import mark_boundaries
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
    def plot_result(self,df1,df2,ch0,ch1,ch2,labels,n):
        fig, ax = plt.subplots(figsize=(10,10))   
        rgb = self.get_rgb(ch0,ch1,ch2)
        rgb = mark_boundaries(rgb,labels,color=(0.8,0.8,0.8))
        ax.imshow(rgb)
        ax.scatter(df1['y'],df1['x'],marker='x',s=5,color='yellow',label='GAPDH')
        ax.scatter(df2['y'],df2['x'],marker='x',s=5,color='purple',label='GBP5')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.opath + self.pfx + self.sfx + f'{n}.png')
    def analyze(self,filters):
        print("Loading dataset...")
        ch0_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch0.tif')
        ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
        ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')
        print("Done.")
        nt,nx,ny = ch0_stack.shape
        for n in range(nt):
            flag = True
            while flag:
                viewer = napari.Viewer()
                viewer.window.resize(2000, 1000)
                viewer.add_image(ch0_stack[n],name='DAPI',colormap='blue',visible=False)
                viewer.add_image(blur(ch1_stack[n],sigma=1),name='GAPDH',colormap='green')
                viewer.add_image(blur(ch2_stack[n],sigma=1),name='GBP5',colormap='red',visible=False)
                napari.run()
                shape = ch1_stack[n].shape
                try:
                    ch1_pts = viewer.layers['GAPDH-features-blob_log'].data
                    ch1_pts = ch1_pts.astype(np.uint16)
                    ch2_pts = viewer.layers['GBP5-features-blob_log'].data
                    ch2_pts = ch2_pts.astype(np.uint16)
                    labels = viewer.layers['Shapes'].to_labels(labels_shape=shape)
                    ch1_pts_labels = np.expand_dims(labels[ch1_pts[:,0],ch1_pts[:,1]],axis=1)
                    ch1_pts = np.concatenate((ch1_pts,ch1_pts_labels),axis=1)
                    df1 = pd.DataFrame({'x':ch1_pts[:, 0],'y':ch1_pts[:, 1],'label':ch1_pts[:,2]})
                    df1.to_csv(self.opath + self.pfx + self.sfx + f'gapdh_{n}.csv')
                    ch2_pts_labels = np.expand_dims(labels[ch2_pts[:,0],ch2_pts[:,1]],axis=1)
                    ch2_pts = np.concatenate((ch2_pts,ch2_pts_labels),axis=1)
                    df2 = pd.DataFrame({'x':ch2_pts[:, 0],'y':ch2_pts[:, 1],'label':ch2_pts[:,2]})
                    df2.to_csv(self.opath + self.pfx + self.sfx + f'gbp5_{n}.csv')
                    self.plot_result(df1,df2,ch0_stack[n],ch1_stack[n],ch2_stack[n],labels,n)
                    flag = False
                except Exception as e:
                    print(e)
                    print('ERROR: Missing layer: Restarting...')

        

