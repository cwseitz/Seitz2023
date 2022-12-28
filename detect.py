from smlm.localize import LOGDetector
from smlm.filters import blur
from pycromanager import Dataset
from skimage.measure import regionprops_table
from skimage.util import map_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib._color_data as mcd
import tifffile

class Detector:
    def __init__(self,ipath,opath,prefix):
        self.ipath = ipath
        self.opath = opath
        self.mask = tifffile.imread(opath + prefix + '_ch0_mask.tif')
    def detect(self,ch1_thres=0.0003,ch2_thres=0.000175,z0=4):
        dataset = Dataset(self.ipath)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        X = X.reshape((nz,nc,nt**2,nx,ny))
        for n in range(nt**2):
            print(f'Processing tile {n}\n')
            mask = self.mask[n]
            mask, table = self.filter_objects(mask)
            centroids_x = table['centroid-0']
            centroids_y = table['centroid-1']
            centroids = np.vstack([centroids_x,centroids_y]).T
            ch1det = LOGDetector(np.array(X[z0,1,n,:,:]),threshold=ch1_thres)
            ch1_blobs = ch1det.detect()
            ch2det = LOGDetector(np.array(X[z0,2,n,:,:]),threshold=ch2_thres)
            ch2_blobs = ch2det.detect()
            ch1_blobs = ch1_blobs.assign(cluster1=None)
            ch1_blobs = ch1_blobs.assign(cluster2=1)
            X_blobs = ch1_blobs[['x','y']].to_numpy()
            n_clusters = centroids.shape[0]
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(centroids)
            pred = kmeans.predict(X_blobs)
            ch1_blobs = ch1_blobs.assign(cluster1=pred)
            self.plot_groups(ch1_blobs,mask)
            
    def filter_objects(self,mask,min_area=5000,max_area=50000,max_ecc=0.75,min_solid=0.9):
        props = ('label', 'area', 'eccentricity','solidity','centroid')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['eccentricity'] < max_ecc) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask, input_labels, output_labels)
        filtered_table = regionprops_table(filtered_mask,properties=props)
        return filtered_mask, filtered_table
    def assign_to_nearest(self,ch1_blobs,ch2_blobs,centroids,mask):
        ch1_assignments = []
        ch2_assignments = []
        for index, row in ch1_blobs.iterrows():
            x0,y0 = row['x'],row['y']
            euclidean = np.sqrt((centroids[:,0] - x0)**2 + (centroids[:,1] - y0)**2)
            idx = np.argmax(euclidean)
            cx, cy = round(centroids[idx,0]), round(centroids[idx,1])
            label = mask[cx,cy]
            ch1_assignments.append(label)
        ch1_blobs = ch1_blobs.assign(cell=ch1_assignments)
        return ch1_blobs
        
    def plot_groups(self,ch1_blobs,mask):
        palette = list(mcd.XKCD_COLORS.values())[::10]
        fig, ax = plt.subplots()
        groups = ch1_blobs['cluster1'].unique()
        ax.imshow(mask,cmap='gray')
        #for group in groups:
        #    this_df = ch1_blobs.loc[ch1_blobs['cluster1'] == group]
        #    X_group = this_df[['x','y']].to_numpy()
        #    clustering = AgglomerativeClustering().fit(X_group)
        #    ch1_blobs.loc[ch1_blobs['cluster1'] == group, 'cluster2'] = clustering.labels_
        groups = ch1_blobs.groupby(['cluster1','cluster2'])
        for i,(name, group) in enumerate(groups):
            ax.scatter(group['y'],group['x'],marker='o',s=3,color=palette[i])
        plt.show()


"""
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
"""
