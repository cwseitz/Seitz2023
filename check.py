import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.io import imread
from smlm.plot import anno_blob
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte, img_as_uint
import tifffile

class Checker:
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

    def filter(self,image):
        blurred = gaussian(image,sigma=1)
        blurred = blurred/blurred.max()
        blurred = img_as_uint(blurred)
        bg = rolling_ball(blurred,radius=5)    
        return blurred-bg  

    def check(self):
 
        ch0_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch0.tif')
        ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
        ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')

        print('Loaded.')

        for n in range(ch0_stack.shape[0]):
            try:
                df1 = pd.read_csv(self.opath + self.pfx + self.sfx + f'gapdh_{n}.csv')
                df2 = pd.read_csv(self.opath + self.pfx + self.sfx + f'gbp5_{n}.csv')
                mask1 = imread(self.opath + self.pfx + self.sfx + f'nuc_mask_{n}.tif')
                mask2 = imread(self.opath + self.pfx + self.sfx + f'cyto_mask_{n}.tif')

                fig, ax = plt.subplots(figsize=(10,10)) 
                rgb = self.get_rgb(ch0_stack[n],3*self.filter(ch1_stack[n]),3*self.filter(ch2_stack[n]))
                rgb = mark_boundaries(rgb,mask1,color=(0.8,0.8,0.8))
                rgb = mark_boundaries(rgb,mask2,color=(0.8,0.8,0.8))
                ax.imshow(rgb)
                anno_blob(ax, df1, color='yellow')
                anno_blob(ax, df2, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(e)
