from smlm.track import LOGDetector
from smlm.filters import blur
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Detector:
	def __init__(self,pfx,opath,sfx='_mxtiled_corrected_stack_'):
		self.pfx = pfx
		self.opath = opath
		self.sfx = sfx
	def detect(self):
		ch1_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch1.tif')
		ch2_stack = tifffile.imread(self.opath + self.pfx + self.sfx + 'ch2.tif')
		nt,nx,ny = ch1_stack.shape
		for n in range(nt):
			print(f'Processing tile {n}\n')
			while True:
				threshold = float(input("Enter GAPDH threshold: "))
				detector = LOGDetector(blur(ch1_stack[n],sigma=1),threshold=threshold)
				detector.detect()
				detector.show()
				plt.show()
				threshold = float(input("Enter GBP5 threshold: "))
				detector = LOGDetector(blur(ch2_stack[n],sigma=1),threshold=threshold)
				detector.detect()
				detector.show()
				plt.show()
				ans = input("Accept (a) or Reject (r): ")
				if ans == 'a':
					break

