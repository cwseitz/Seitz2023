import matplotlib.pyplot as plt
import numpy as np
import napari 
import tifffile
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.util import img_as_int
from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu
from skimage.segmentation import mark_boundaries
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label

class TrainDataGenerator:
    def __init__(self,dir,file,file2,n0=0):
        self.dir = dir
        self.file = file
        self.file2 = file2
        self.stack = tifffile.imread(dir+file)
        self.ch1 = tifffile.imread(dir+file2)
        self.pfx = file.split('.')[0]
        self.n0 = n0
    def mark_boundaries(self):
        nt,nx,ny = self.stack.shape
        for n in range(n0,nt):
            image = self.stack[n]
            image = resize(image,(256,256))
            image = gaussian(image,sigma=1)
            viewer = napari.Viewer()
            viewer.window.resize(2000, 1000)
            thresh = threshold_otsu(image)
            mask0 = np.zeros_like(image,dtype=np.bool)
            mask0[image > thresh] = True
            mask0 = remove_small_objects(mask0)
            viewer.add_image(image,name='Membrane',colormap='yellow')
            s = viewer.add_labels(mask0,name='Mask',visible=True,opacity=0.3)
            s.brush_size = 2
            s.selected_label = 0
            napari.run()
            this_shape = image.shape
            mask = viewer.layers['Mask'].data
            mask = mask.astype(np.uint16)
            mask_dir = self.dir + 'masks/'
            #imsave(mask_dir + self.pfx + f'_mask_{n}.tif',mask)


n0 = 50
dir = '/research3/shared/cwseitz/Analysis/221218-Hela-IFNG-16h-2_1/Train/' 
file = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch2.tif'
file2 = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch1.tif'
train_generator = TrainDataGenerator(dir,file,file2,n0=n0)
train_generator.mark_boundaries()
