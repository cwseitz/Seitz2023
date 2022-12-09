import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.util import img_as_ubyte, map_array
from skimage.filters import threshold_otsu, gaussian, threshold_multiotsu
from skimage.segmentation import watershed, clear_border, mark_boundaries
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.io import imread, imsave
from skimage.measure import regionprops_table, label
from scipy import ndimage as ndi

class Segmenter:
	def __init__(self,pfx,opath,sfx='_mxtiled_corrected_'):
		self.pfx = pfx
		self.opath = opath
		self.sfx = sfx
	def stack2tiled(self,stack):
		nt,nx,ny = (10,1844,1844)
		stack = stack.reshape((nt,nt,nx,ny))
		stack = stack.swapaxes(1,2)
		tiled = stack.reshape((nt*nx,nt*ny))
		return tiled
	def tiled2stack(self,tiled):
		stack = tiled.reshape((100,1844,1844))
		return stack
	def segment_nucleus(self,image,filters):
		image = gaussian(image,sigma=2)
		thresholds = threshold_multiotsu(image, classes=3)
		mask = image > thresholds[1]
		mask = self.filter_objects(label(mask),**filters)
		mask = clear_border(mask)
		#mask = erosion(mask)
		distance = ndi.distance_transform_edt(mask)
		coords = peak_local_max(distance, min_distance=100)
		_mask = np.zeros(mask.shape, dtype=bool)
		_mask[tuple(coords.T)] = True
		markers, _ = ndi.label(_mask)
		mask = watershed(-distance,markers,mask=mask)
		return mask, coords

	def segment_cytoplasm(self,ch0,ch1,coords,filters):
		image = gaussian(ch1,sigma=5)
		thresholds = threshold_multiotsu(image, classes=3)
		mask = image > 0.0055
		mask = self.filter_objects(label(mask),**filters)
		mask = clear_border(mask)
		#mask = erosion(mask)
		distance = ndi.distance_transform_edt(mask)
		_mask = np.zeros(mask.shape, dtype=bool)
		_mask[tuple(coords.T)] = True
		markers, _ = ndi.label(_mask)
		mask = watershed(-distance,markers,mask=mask)
		return image, mask

	def segment(self,nuc_filters,cyto_filters,plot=True):
		ch0 = np.load(self.opath + self.pfx + self.sfx + 'ch0.npz')['ch0']
		ch0_stack = self.tiled2stack(ch0)
		nt,nx,ny = ch0_stack.shape
		ch0_mask_stack = np.zeros((100,1844,1844),dtype=np.uint8)
		ch1 = np.load(self.opath + self.pfx + self.sfx + 'ch1.npz')['ch1']
		ch1_stack = self.tiled2stack(ch1)
		ch1_mask_stack = np.zeros((100,1844,1844),dtype=np.uint8)
		for n in range(nt):
			ch0_mask_stack[n], coords =\
			self.segment_nucleus(ch0_stack[n],nuc_filters)
			combined, ch1_mask_stack[n] =\
			self.segment_cytoplasm(ch0_stack[n],ch1_stack[n],coords,cyto_filters)
			if plot:
				fig, ax = plt.subplots(1,2)
				im1 = mark_boundaries(50*ch0_stack[n],ch0_mask_stack[n])
				im2 = mark_boundaries(50*combined,ch1_mask_stack[n])
				#ax[0,0].imshow(ch0_mask_stack[n])
				#ax[0,1].imshow(ch1_mask_stack[n])
				ax[0].imshow(im1)
				ax[1].imshow(im2)
				plt.tight_layout()
				plt.show()

		# ch0_mask_tiled = self.stack2tiled(ch0_mask_stack)
		# ch0_mask_tiled = self.ch0_mask_tiled.astype(np.uint16)
		# return ch0_mask_tiled

	def filter_objects(self,mask,min_area=5000,max_area=50000,max_ecc=0.75,min_solid=0.9):
		props = ('label', 'area', 'eccentricity','solidity')
		print(min_area,max_area,max_ecc,min_solid)
		table = regionprops_table(mask,properties=props)
		condition = (table['area'] > min_area) &\
					(table['area'] < max_area) &\
					(table['eccentricity'] < max_ecc) &\
					(table['solidity'] > min_solid)
		input_labels = table['label']
		output_labels = input_labels * condition
		filtered_mask = map_array(mask, input_labels, output_labels)
		return filtered_mask
