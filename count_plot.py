from pipeline import Pipeline
from count import SpotCounts
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'221218-Hela-IFNG-4h-1_1',
'221218-Hela-IFNG-16h-2_1',
]

with open('config.json', 'r') as f:
    config = json.load(f)

counts = []
for prefix in prefixes:
    sc = SpotCounts(config['datapath'],config['analpath'],prefix)
    spots, count_matrix = sc.count_matrix(plot=False)
    count_matrix = count_matrix.loc[count_matrix['gapdh'] > 100]
    counts.append(count_matrix)
time_df = pd.concat(counts)


######################
# Histograms
######################

fig, ax = plt.subplots(2,2,figsize=(5,3),sharex=False,sharey=True)
ch1_bins = 10
ch2_bins = 10
ax[0,0].hist(counts[0]['gapdh'],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,0].set_ylabel('Number of cells')
ax[0,0].set_title(f'4h (N={counts[0].shape[0]})')
ax[0,0].legend()
ax[1,0].hist(counts[0]['gbp5'],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,0].set_xlabel('mRNA Counts')
ax[1,0].set_ylabel('Number of cells')
ax[1,0].legend()
ax[0,1].hist(counts[1]['gapdh'],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,1].set_title(f'16h (N={counts[1].shape[0]})')
ax[0,1].legend()
ax[1,1].hist(counts[1]['gbp5'],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,1].set_xlabel('mRNA Counts')
ax[1,1].legend()
plt.tight_layout()
plt.show()

######################
# Time plot
######################

avg_gapdh_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gapdh'].mean()
avg_gapdh_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gapdh'].mean()
std_gapdh_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gapdh'].std()
std_gapdh_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gapdh'].std()

gbp5_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gbp5']
gbp5_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gbp5']
avg_gbp5_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gbp5'].mean()
avg_gbp5_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gbp5'].mean()
std_gbp5_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gbp5'].std()
std_gbp5_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gbp5'].std()

t, pval = ttest_ind(gbp5_4h,gbp5_16h)
print(pval)
time = [4,16] #hours
x = np.linspace(0,1,2)
avgs = np.array([avg_gbp5_4h,avg_gbp5_16h])
yerr = np.array([std_gbp5_4h,std_gbp5_16h])/np.sqrt(200)
width = 0.5
labels = ['4h','16h']
fig, ax = plt.subplots(figsize=(2,3))
ax.set_xticks(x,labels,weight='bold')
ax.set_ylabel('GBP5 mRNA',fontsize=12,weight='bold')
ax.bar(x,avgs,width,yerr=yerr,color='red',capsize=5)
plt.tight_layout()
plt.show()


