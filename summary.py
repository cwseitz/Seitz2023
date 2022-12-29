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

class Summarize:
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
        return image

    def check(self):

        ch0_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch0.tif')
        ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
        ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')
        print('Loaded.')
        for n in range(ch0_stack.shape[0]):
            try:
                mask1 = imread(self.opath + self.pfx + self.sfx + f'nuc_mask_{n}.tif')
                mask2 = imread(self.opath + self.pfx + self.sfx + f'cyto_mask_{n}.tif')
                df1 = pd.read_csv(self.opath + self.pfx + self.sfx + f'gapdh_{n}.csv',index_col=0)
                df2 = pd.read_csv(self.opath + self.pfx + self.sfx + f'gbp5_{n}.csv',index_col=0) 

                #df1 = df1.drop(['label','intensity'],errors='ignore')
                #df1['label'] = mask2[df1['x'],df1['y']]
                #df1['intensity'] = ch1_stack[n][df1['x'],df1['y']] 
                #df2 = df2.drop(['label','intensity'],errors='ignore')
                #df2['label'] = mask2[df2['x'],df2['y']]
                #df2['intensity'] = ch2_stack[n][df2['x'],df2['y']]
                #df1.to_csv(self.opath + self.pfx + self.sfx + f'gapdh_{n}.csv')
                #df2.to_csv(self.opath + self.pfx + self.sfx + f'gbp5_{n}.csv')

                df1 = df1.loc[df1['label'] != 0]
                df2 = df2.loc[df2['label'] != 0]
                vals, bins = np.histogram(df1['intensity'],bins=10)
                print(f'Found all files for tile {n}') 
                fig, ax = plt.subplots(1,2,figsize=(8,6)) 
                rgb = self.get_rgb(ch0_stack[n],3*self.filter(ch1_stack[n]),3*self.filter(ch2_stack[n]))
                rgb = mark_boundaries(rgb,mask1,color=(0.8,0.8,0.8))
                rgb = mark_boundaries(rgb,mask2,color=(0.8,0.8,0.8))
                ax[0].imshow(rgb)
                anno_blob(ax[0], df1, color='yellow')
                anno_blob(ax[0], df2, color='red')
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[0].legend()
                ax[1].plot(bins[:-1],vals,color='blue')
                plt.tight_layout()
                plt.show()

            except Exception as e:
               print(f'Missing files for tile {n}') 
               
               
                       
    def filter_objects(self,mask,min_area=5000,max_area=50000,max_ecc=0.75,min_solid=0.9):
        props = ('label', 'area', 'eccentricity','solidity','centroid')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['eccentricity'] < max_ecc) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask, input_labels, output_labels)
        filtered_table = regionprops_table(filtered_mask,properties=props)
        return filtered_mask, filtered_table
        
    def plot_groups(self,ch1_blobs,mask):
        palette = list(mcd.XKCD_COLORS.values())[::10]
        fig, ax = plt.subplots()
        groups = ch1_blobs['cluster1'].unique()
        ax.imshow(mask,cmap='gray')
        #for group in groups:
        #    this_df = ch1_blobs.loc[ch1_blobs['cluster1'] == group]
        #    X_group = this_df[['x','y']].to_numpy()
        #    clustering = AgglomerativeClustering().fit(X_group)
        #    ch1_blobs.loc[ch1_blobs['cluster1'] == group, 'cluster2'] = clustering.labels_
        groups = ch1_blobs.groupby(['cluster1','cluster2'])
        for i,(name, group) in enumerate(groups):
            ax.scatter(group['y'],group['x'],marker='o',s=3,color=palette[i])
        plt.show()
