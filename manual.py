import matplotlib.pyplot as plt
import numpy as np
import tifffile
import math
import pandas as pd
from skimage.segmentation import mark_boundaries
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.util import img_as_ubyte, img_as_uint
from skimage.io import imsave
from smlm.track import LOGDetector
from smlm.filters import blur, boxcar
from smlm.plot import anno_blob
import napari 

class Analyzer:
    def __init__(self,pfx,opath,sfx='_mxtiled_corrected_stack_',diagnostic=False):
        self.pfx = pfx
        self.opath = opath
        self.sfx = sfx
        self.diagnostic = diagnostic
    def get_rgb(self,ch0,ch1,ch2):
        ch0_max = ch0.max()
        ch1_max = ch1.max()
        ch2_max = ch2.max()
        rgb = np.dstack((ch2/ch2_max,ch1/ch1_max,ch0/ch0_max))
        return rgb
    def plot_result(self,df1,df2,ch0,ch1,ch2,labels,labels2,n):
        fig, ax = plt.subplots(figsize=(10,10)) 
        rgb = self.get_rgb(ch0,ch1,ch2)
        rgb = mark_boundaries(rgb,labels,color=(0.8,0.8,0.8))
        rgb = mark_boundaries(rgb,labels2,color=(0.8,0.8,0.8))
        ax.imshow(rgb)
        anno_blob(ax, df1, color='yellow')
        anno_blob(ax, df2, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        plt.tight_layout()
        plt.show()
    def filter(self,image):
        blurred = gaussian(image,sigma=1)
        blurred = blurred/blurred.max()
        blurred = img_as_uint(blurred)
        bg = rolling_ball(blurred,radius=5)    
        return blurred-bg  
    def analyze(self,n0=0):
        print("Loading dataset...")
        ch0_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch0.tif')
        ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
        ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')
        print("Done.")
        nt,nx,ny = ch0_stack.shape
        for n in range(n0,nt):
            flag = True
            while flag:
                viewer = napari.Viewer()
                viewer.window.resize(2000, 1000)
                viewer.add_image(ch0_stack[n],name='DAPI',colormap='blue',visible=False)
                viewer.add_image(self.filter(ch1_stack[n]),name='GAPDH',colormap='green')
                viewer.add_image(self.filter(ch2_stack[n]),name='GBP5',colormap='yellow',visible=False)
                viewer.add_shapes(name='Nucleus')
                viewer.add_shapes(name='Cytoplasm')
                napari.run()
                shape = ch1_stack[n].shape
                ans = input("Save? (y/n): ")
                if ans == 'y':
                    try:
                        ch1_pts = viewer.layers['GAPDH-features-blob_log'].data
                        ch1_pts = ch1_pts.astype(np.uint16)
                        ch2_pts = viewer.layers['GBP5-features-blob_log'].data
                        ch2_pts = ch2_pts.astype(np.uint16)
                        labels = viewer.layers['Nucleus'].to_labels(labels_shape=shape).astype(np.uint16)
                        labels2 = viewer.layers['Cytoplasm'].to_labels(labels_shape=shape).astype(np.uint16)
                        ch1_pts_labels = np.expand_dims(labels[ch1_pts[:,0],ch1_pts[:,1]],axis=1)
                        ch1_pts_int = np.expand_dims(ch1_stack[n][ch1_pts[:,0],ch1_pts[:,1]],axis=1)
                        ch1_pts = np.concatenate((ch1_pts,ch1_pts_labels,ch1_pts_int),axis=1)
                        df1 = pd.DataFrame({'x':ch1_pts[:, 0],'y':ch1_pts[:, 1],'label':ch1_pts[:,2],'intensity':ch1_pts[:,3]})
                        df1.to_csv(self.opath + self.pfx + self.sfx + f'gapdh_{n}.csv')
                        ch2_pts_labels = np.expand_dims(labels[ch2_pts[:,0],ch2_pts[:,1]],axis=1)
                        ch2_pts_int = np.expand_dims(ch2_stack[n][ch2_pts[:,0],ch2_pts[:,1]],axis=1)
                        ch2_pts = np.concatenate((ch2_pts,ch2_pts_labels,ch2_pts_int),axis=1)
                        df2 = pd.DataFrame({'x':ch2_pts[:, 0],'y':ch2_pts[:, 1],'label':ch2_pts[:,2],'intensity':ch2_pts[:,3]})
                        df2.to_csv(self.opath + self.pfx + self.sfx + f'gbp5_{n}.csv')
                        if self.diagnostic:
                            self.plot_result(df1,df2,ch0_stack[n],self.filter(ch1_stack[n]),
                                             self.filter(ch2_stack[n]),labels,labels2,n)
                        imsave(self.opath + self.pfx + self.sfx + f'nuc_mask_{n}.tif', labels)
                        imsave(self.opath + self.pfx + self.sfx + f'cyto_mask_{n}.tif', labels2)
                        flag = False
                    except Exception as e:
                        print(e)
                        print('ERROR: Restarting...')
                else:
                    flag = False

        

