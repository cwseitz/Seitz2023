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
    spots, count_matrix = sc.count_matrix(plot=False,z=5)
    #id_check = count_matrix.loc[count_matrix['gapdh'] < 100, 'cell_id']
    count_matrix = count_matrix.loc[count_matrix['gapdh'] > 100]
    counts.append(count_matrix)
time_df = pd.concat(counts)


######################
# Histograms
######################

fig, ax = plt.subplots(2,3,figsize=(6,3),sharex=False,sharey=True)
ch1_bins = np.arange(0,1000,100)
ch2_bins = 8
ax[0,0].hist(counts[0]['gapdh'],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,0].set_ylabel('Number of cells')
ax[0,0].set_title(f'2h (N={counts[0].shape[0]})')
ax[0,0].legend()
ax[1,0].hist(counts[0]['gbp5'],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,0].set_xlabel('mRNA Counts')
ax[1,0].set_ylabel('Number of cells')
ax[1,0].legend()
ax[0,1].hist(counts[1]['gapdh'],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,1].set_title(f'4h (N={counts[1].shape[0]})')
ax[0,1].legend()
ax[1,1].hist(counts[1]['gbp5'],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,1].set_xlabel('mRNA Counts')
ax[1,1].legend()
ax[0,2].hist(counts[2]['gapdh'],bins=ch1_bins,density=False,color='black',label='GAPDH')
ax[0,2].set_title(f'16h (N={counts[2].shape[0]})')
ax[0,2].legend()
ax[1,2].hist(counts[2]['gbp5'],bins=ch2_bins,density=False,color='blue',label='GBP5')
ax[1,2].set_xlabel('mRNA Counts')
ax[1,2].legend()
plt.tight_layout()
plt.show()

######################
# Time plot
######################

avg_gapdh_2h = time_df.loc[time_df['grid'] == '221130-Hela-IFNG-2h-1_1','gapdh'].mean()
avg_gapdh_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gapdh'].mean()
avg_gapdh_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gapdh'].mean()
std_gapdh_2h = time_df.loc[time_df['grid'] == '221130-Hela-IFNG-2h-1_1','gapdh'].std()
std_gapdh_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gapdh'].std()
std_gapdh_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gapdh'].std()
avg_gbp5_2h = time_df.loc[time_df['grid'] == '221130-Hela-IFNG-2h-1_1','gbp5'].mean()
avg_gbp5_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gbp5'].mean()
avg_gbp5_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gbp5'].mean()
std_gbp5_2h = time_df.loc[time_df['grid'] == '221130-Hela-IFNG-2h-1_1','gbp5'].std()
std_gbp5_4h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-4h-1_1','gbp5'].std()
std_gbp5_16h = time_df.loc[time_df['grid'] == '221218-Hela-IFNG-16h-2_1','gbp5'].std()

fig, ax = plt.subplots()
time = [2,4,16] #hours
avg_gbp5_4h_fc = np.log2((avg_gbp5_4h-avg_gbp5_2h)/avg_gbp5_2h)
avg_gbp5_16h_fc = np.log2((avg_gbp5_16h-avg_gbp5_2h)/avg_gbp5_2h)
avg_gbp5_2h = 0
avg = [avg_gbp5_2h,avg_gbp5_4h,avg_gbp5_16h]
stderr = [std_gbp5_2h/np.sqrt(200),std_gbp5_4h/np.sqrt(200),std_gbp5_16h/np.sqrt(200)]
ax.errorbar(time,avg,yerr=stderr,color='black',capsize=6)
ax.set_xlabel('Time (hours)',size=12,weight='bold')
ax.set_ylabel('log2FC',size=12,weight='bold')
plt.show()


