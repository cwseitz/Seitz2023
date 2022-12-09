import matplotlib.pyplot as plt
import numpy as np
import tifffile
import math
from smlm.track import LOGDetector
from smlm.filters import blur
from smlm.plot import anno_blob
from skimage.feature import peak_local_max
from skimage.util import img_as_ubyte, map_array
from skimage.filters import threshold_otsu, gaussian, threshold_multiotsu, laplace
from skimage.segmentation import watershed, clear_border, mark_boundaries
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.io import imread, imsave
from skimage.measure import regionprops_table, label
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform

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
            print(f'Processing tile {n}')
            self.plot_raw(ch0_stack[n],ch1_stack[n],ch2_stack[n])
            ans = input("Process? (y/n): ")
            if ans == 'y':
                i = 0
                while True:
                    if i == 0:
                        ch1_thr = ch1_dft
                        ch2_thr = ch2_dft
                    else:
                        ch1_thr = float(input("Threshold ch1: "))
                        ch2_thr = float(input("Threshold ch2: "))
                    i += 1
                    mask,coords = self.segment_nucleus(ch0_stack[n],filters)
                    rgb = self.get_rgb(ch0_stack[n],ch1_stack[n],ch2_stack[n])
                    self.segment_cells(rgb)
                    df0 = self.detect_nuclei(ch0_stack[n])
                    df1, df2 = self.detect_spots(ch1_stack[n],ch2_stack[n],ch1_thr,ch2_thr)
                    self.plot_result(ch0_stack[n],ch1_stack[n],ch2_stack[n],mask,df0,df1,df2)
                    ans = input("Accept (a) or Reject (r): ")
                    if ans == 'a':
                        break
    
    def detect_nuclei(self,ch0):
        det0 = LOGDetector(ch0,min_sigma=20,max_sigma=60,num_sigma=3,threshold=0.001)
        det0.detect()
        return det0.blobs_df
        
    def detect_spots(self,ch1,ch2,ch1_thr,ch2_thr):
        det1 = LOGDetector(blur(ch1,sigma=1),threshold=ch1_thr)
        det1.detect()
        det2 = LOGDetector(blur(ch2,sigma=1),threshold=ch2_thr)
        det2.detect()
        return det1.blobs_df, det2.blobs_df
        
    def segment_cells(self,rgb):
        x = laplace(rgb[:,:,0])
        plt.imshow(x)
        plt.show()
                    
    def segment_nucleus(self,image,filters):
        image = gaussian(image,sigma=2)
        thresholds = threshold_multiotsu(image, classes=3)
        mask = image > thresholds[1]
        mask = self.filter_objects(label(mask),**filters)
        mask = clear_border(mask)
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, min_distance=100)
        _mask = np.zeros(mask.shape, dtype=bool)
        _mask[tuple(coords.T)] = True
        markers, _ = ndi.label(_mask)
        mask = watershed(-distance,markers,mask=mask)
        return mask, coords


    def filter_objects(self,mask,min_area=5000,max_area=50000,max_ecc=0.75,min_solid=0.9):
        props = ('centroid','label', 'area', 'eccentricity','solidity')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['eccentricity'] < max_ecc) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask, input_labels, output_labels)
        return filtered_mask
