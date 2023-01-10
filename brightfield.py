from pycromanager import Dataset
from gptie import *
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json



prefixes = [
'221120-8 well glass slide-Hela-0h-1_1'
]

path = '/research3/shared/cwseitz/Archive/Data'
this_path = path + '/' + prefixes[0]
dataset = Dataset(this_path)
X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
ch3 = X[:,-1,5,5,:,:]
z_vec = np.arange(-4,5)*1e-6 #microns
lambd = 520e-9 #nm
ps = 108.3e-9 #nm
zfocus = 4
Nsl = 100
plt.imshow(ch3[0])
plt.show()
ch3 = ch3[:,1000:1500,1000:1500].astype(np.float)

ch3 = np.moveaxis(ch3,0,-1)
phase = GP_TIE(ch3,z_vec,lambd,ps,zfocus)
fig, ax = plt.subplots(1,2)
ax[0].imshow(ch3[:,:,0],cmap='gray')
ax[1].imshow(phase)
plt.show()

    
