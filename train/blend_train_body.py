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

class Blender:
    def __init__(self,stack_path,output_dir,ch0_file,ch1_file):
        self.stack_path = stack_path
        self.output_dir = output_dir
        self.ch0 = tifffile.imread(stack_path+ch0_file)
        self.ch1 = tifffile.imread(stack_path+ch1_file)
        self.pfx = ch1_file.split('ch1')[0]
    def resize_and_blend(self,ch0,ch1,new_size=(256,256)):
        ch0 = resize(ch0,new_size)
        ch1 = resize(ch1,new_size)
        r = ch0.max()/ch1.max()
        ch1 *= r
        blended = 0.4*ch0 + 0.6*ch1
        blended = gaussian(blended,sigma=1)
        blended = rolling_ball(blended,radius=50)
        blended = img_as_int(2*blended)
        return blended
    def blend_stacks(self):
        nt,nx,ny = self.ch0.shape
        blended_stack = np.zeros((nt,256,256),dtype=np.int16)
        for n in range(nt):
            print(f'Blending image: {n}')
            blended_stack[n] = self.resize_and_blend(self.ch0[n],self.ch1[n])
            labels = viewer.layers['Mask'].to_labels(labels_shape=this_shape).astype(np.uint16)
        imsave(self.output_dir + self.pfx + 'blended.tif',blended_stack)


stack_path = '/research3/shared/cwseitz/Analysis/221218-Hela-IFNG-16h-2_1/'
output_dir = '/research3/shared/cwseitz/Analysis/221218-Hela-IFNG-16h-2_1/' 
ch0 = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch0.tif'
ch1 = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch1.tif'
blender = Blender(stack_path,output_dir,ch0,ch1)
blender.blend_stacks()
