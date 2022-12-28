import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.segmentation import mark_boundaries

def check_segments(prefix,stack_path):
    ch0_mask_path = stack_path + prefix + '_mxtiled_corrected_ch0_mask.tif'
    ch1_mask_path = stack_path + prefix + '_mxtiled_corrected_ch1_mask.tif'
    ch0_mask = tifffile.imread(ch0_mask_path)
    ch1_mask = tifffile.imread(ch1_mask_path)
    ch0_path = stack_path + prefix + '_mxtiled_corrected_stack_ch0.tif'
    ch1_path = stack_path + prefix + '_mxtiled_corrected_stack_ch1.tif'
    ch0 = tifffile.imread(ch0_path)
    ch1 = tifffile.imread(ch1_path)
    nt,nx,ny = ch0.shape
    for n in range(nt):
        fig, ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(15,8))
        ax[0].imshow(mark_boundaries(30*ch0[n],ch0_mask[n]))
        ax[1].imshow(mark_boundaries(100*ch1[n],ch1_mask[n]))
        plt.tight_layout()
        plt.show()
        
prefix = '221218-Hela-IFNG-16h-2_1'
stack_path = '/research3/shared/cwseitz/Analysis/' + prefix + '/'
check_segments(prefix,stack_path)
