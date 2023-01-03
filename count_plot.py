from pipeline import Pipeline
from count import SpotCounts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'221130-Hela-IFNG-2h-1_1',
'221218-Hela-IFNG-4h-1_1',
#'221130-Hela-IFNG-8h-1_1',
'221218-Hela-IFNG-16h-2_1',
]

with open('config.json', 'r') as f:
    config = json.load(f)

counts = []
for prefix in prefixes:
    sc = SpotCounts(config['datapath'],config['analpath'],prefix)
    count_matrix = sc.count_matrix(plot=False,z=5)
    counts.append(count_matrix)

ch1_avg = []; ch2_avg = []
for count_matrix in counts:
    ch1_avg.append(np.mean(count_matrix[:,0]))
    ch2_avg.append(np.mean(count_matrix[:,1]))

fig, ax = plt.subplots(2,3,figsize=(6,3),sharex=False,sharey=True)
ch1_bins = np.arange(0,1000,100)
ch2_bins = 8
ax[0,0].hist(counts[0][:,0],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,0].set_ylabel('Number of cells')
ax[0,0].set_title(f'2h (N={counts[0].shape[0]})')
ax[0,0].legend()
ax[1,0].hist(counts[0][:,1],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,0].set_xlabel('mRNA Counts')
ax[1,0].set_ylabel('Number of cells')
ax[1,0].legend()
ax[0,1].hist(counts[1][:,0],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,1].set_title(f'4h (N={counts[1].shape[0]})')
ax[0,1].legend()
ax[1,1].hist(counts[1][:,1],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,1].set_xlabel('mRNA Counts')
ax[1,1].legend()
ax[0,2].hist(counts[2][:,0],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,2].set_title(f'16h (N={counts[2].shape[0]})')
ax[0,2].legend()
ax[1,2].hist(counts[2][:,1],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,2].set_xlabel('mRNA Counts')
ax[1,2].legend()
#ax[1,1].set_xlim([0,1])
plt.tight_layout()
plt.show()



