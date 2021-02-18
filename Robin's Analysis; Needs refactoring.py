# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:12:56 2021

@author: Peter
"""


#Try LOG on larger cropped image. 



#Using Laplacian of Gaussian to identify circles in overlapping blobs
#Works the best so far!

#Load File
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io, measure
import cv2


#Load image. 
#filename_load = '200nm Au NP 20x 0.50 NA collecting obj - 2.2 Cropped.tif'  
filename_load = '200nm Au NP 20x 0.50 NA collecting obj - 2.2 Cropped; larger.tif'  
file_extension = "C:/Users/Peter/Desktop/Robin's Images/"



from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import statistics



image = cv2.imread(file_extension + filename_load, 0) 
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, min_sigma = 1, max_sigma=50, num_sigma=50, threshold=0.06, overlap = 0.1)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) # Compute radii in the 3rd column.


blobs_list = [blobs_log, image]
colors = ['yellow', 'lime']
titles = ['Laplacian of Gaussian', 'Original']

sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        try:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
            ax[idx].add_patch(c)
            ax[idx].set_axis_off()
        except:
            continue
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 15))
plt.imshow(image)
plt.show()   













#blobs_log
#Use coordinates of 'blobs_log' to find n closest points. 


def distance(p1,p2):
    """Euclidean distance between two points."""
    from math import hypot
    x1,y1 = p1
    x2,y2 = p2
    return hypot(x2 - x1, y2 - y1)




blobs_log_df = pd.DataFrame(blobs_log).iloc[:,:2]
coords = [tuple(x) for x in blobs_log_df.to_numpy()] #convert blobs_log to list of tuples. 


mean_n = []
median_n = []
std_n = []
for n in range(1,21): #cycle over range of n values. #1-20. 

    top_n_dist = []
    for current_coord in coords: #Cycle through each coordinate to compare all else to. 
        
        current_dist = []
        for coord in coords: #Cycle through all else to compare to 'current_coord'
            current_dist.append(distance(current_coord, coord))
            
        top_n_dist.extend(sorted(current_dist)[1:n+1]) #after distances calculated for current_coord, only select n to save. 
        
    top_n_dist_df = pd.DataFrame(top_n_dist)
    sns.distplot(top_n_dist_df, norm_hist = True)
    # top_n_dist_df.plot(kind = "hist", density = True, alpha = 0.65, bins = 15) # change density to true, because KDE uses density
    # top_n_dist_df.plot(kind = "kde")
    plt.title("n = " + str(n))
    plt.xlim(0,100)
    plt.ylim(0, 0.2)
    plt.text(60, 0.18, "Mean = " + str(round(float(np.mean(top_n_dist_df)), 2)))
    plt.text(60, 0.16, "Median = " + str(round(float(np.median(top_n_dist_df)), 2)))
    plt.text(60, 0.14, "Stdev = " + str(round(float(np.std(top_n_dist_df)), 2)))
    plt.show()
    
    mean_n.append(round(float(np.mean(top_n_dist_df)), 2))
    median_n.append(round(float(np.median(top_n_dist_df)), 2))
    std_n.append(round(float(np.std(top_n_dist_df)), 2))
