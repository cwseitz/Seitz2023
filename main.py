from pipeline import Pipeline
from count import SpotCounts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'221130-Hela-IFNG-2h-1_1',
#'221218-Hela-IFNG-4h-1_1',
'221130-Hela-IFNG-8h-1_1',
'221218-Hela-IFNG-16h-2_1',
]

with open('config.json', 'r') as f:
    config = json.load(f)
    
#for prefix in prefixes:
    #print("Processing " + prefix)
    #pipe = Pipeline(config,prefix)
    #pipe.tile()
    #pipe.basic_correct()
    #pipe.apply_nucleus_model()
    #pipe.apply_cell_model()
    #pipe.detect_spots()
    #pipe.segment_cells()

counts = [] #time, cell, gene
for prefix in prefixes:
    sc = SpotCounts(config['datapath'],config['analpath'],prefix)
    count_matrix = sc.count_matrix()
    counts.append(count_matrix)


ch1_avg = []; ch2_avg = []
for count_matrix in counts:
    ch1_avg.append(np.mean(count_matrix[:,0]))
    ch2_avg.append(np.mean(count_matrix[:,1]))

fig, ax = plt.subplots()
#ax.plot(ch1_avg,color='black')
ax.plot(ch2_avg,color='purple')
plt.show()


fig, ax = plt.subplots(2,3,figsize=(6,3),sharex=False,sharey=False)
ch1_bins = np.arange(0,10000,1000)
ch2_bins = np.arange(0,500,50)
ax[0,0].hist(counts[0][:,0],bins=ch1_bins,density=True,color='black',label='GAPDH')
ax[0,0].set_ylabel('PDF')
ax[0,0].set_title('2h')
ax[0,0].legend()
ax[1,0].hist(counts[0][:,1],bins=ch2_bins,density=True,color='purple',label='GBP5')
ax[1,0].set_xlabel('Counts')
ax[1,0].set_ylabel('PDF')
ax[1,0].legend()
ax[0,1].hist(counts[1][:,0],bins=ch1_bins,density=True,color='black',label='GAPDH')
ax[0,1].set_title('8h')
ax[0,1].legend()
ax[1,1].hist(counts[1][:,1],bins=ch2_bins,density=True,color='purple',label='GBP5')
ax[1,1].set_xlabel('Counts')
ax[1,1].legend()
ax[0,2].hist(counts[2][:,0],bins=ch1_bins,density=True,color='black',label='GAPDH')
ax[0,2].set_title('16h')
ax[0,2].legend()
ax[1,2].hist(counts[2][:,1],bins=ch2_bins,density=True,color='purple',label='GBP5')
ax[1,2].set_xlabel('Counts')
ax[1,2].legend()
plt.tight_layout()
plt.show()


