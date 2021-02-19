# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:25:59 2021

@author: Peter
"""

import cv2
from skimage.color import rgb2gray
from skimage.feature import blob_log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from math import hypot





def extract_LOG_coordinates(image): 
    """
    Extracts Coordinates of Nanoparticle Centers. 
    
    Parameters
    ----------
    image : Array of uint8
        (.tif) image size 387 x 535, showing Nanoparticles dispersed across surface. 

    Returns
    -------
    laplace_coords : Array of float64
        where columns = [X, Y, radius]

    """
    
    image_gray = rgb2gray(image)
    
    laplace_coords = blob_log(image_gray, min_sigma = 1, max_sigma=50, num_sigma=50, threshold=0.06, overlap = 0.1)
    laplace_coords[:, 2] = laplace_coords[:, 2] * sqrt(2) # Compute radii in the 3rd column.

    #blobs_count = len(laplace_coords) #Number of total blobs. 

    return laplace_coords




def plot_overlaid_coordinates(laplace_coords, image):
    """
    Plots Original Image vs. Original Image with overlaid locations of nanoparticles, identified by laplace_coords

    Parameters
    ----------
    laplace_coords : Array of float64
        where columns = [X, Y, radius]
    image : Array of uint8
        Original image loaded. Used to make sure ID'd coordinates are in correct location.

    Returns
    -------
    None.

    """
    
    #Plot Subplots
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    
    ax1.imshow(image)
    ax2.imshow(image)
    
    ax1.set_title("Original")    
    ax2.set_title("Laplacian of Gaussian")
    
    #Axes = Pixels
    # ax1.set_axis_off()
    # ax2.set_axis_off()


    for blob in laplace_coords:
        y, x, r = blob
        c = plt.Circle((x, y), r, color="yellow", linewidth=1, fill=False)
        ax2.add_patch(c)
    
    plt.tight_layout()
    plt.show()
    
    



def euclidean_distance(p1,p2):
    """
    Calculates Euclidean distance between two points.

    Parameters
    ----------
    p1 : int
        First point to use in distance calculation.
    p2 : int
        Second point to use in distance calculation. 

    Returns
    -------
    float
        Represents the Euclidean distance from the origin for the inputs.

    """
    
    x1,y1 = p1
    x2,y2 = p2
    return hypot(x2 - x1, y2 - y1)











def neighboring_distance(laplace_coords, n_neighbors, max_distance = 50):
    """
    Creates single column dataframe of euclidean distances between points, considering 'n_neighbors'. Can optionally include a max_distance.

    Parameters
    ----------
    laplace_coords : Array of float64
        where columns = [X, Y, radius]
        
    n_neighbors : int
        # neighbors to consider per coordinate in laplace_coords. 

    Returns
    -------
    top_n_dist_df : DataFrame
        Saved Euclidean distance values for n_neighbors per coordinate. 

    """

    laplace_coords_df = pd.DataFrame(laplace_coords).iloc[:,:2]
    coords = [tuple(x) for x in laplace_coords_df.to_numpy()] #convert laplace_coords to list of tuples. 
    
    top_n_dist = []
    for current_coord in coords: #Cycle through each coordinate to compare all else to. 
        
        current_dist = []
        for coord in coords: #Cycle through all else to compare to 'current_coord'
            distance = euclidean_distance(current_coord, coord)
            if distance < max_distance: #Implments a max distance to consider. 
                current_dist.append(distance)
            else: 
                continue
            
        top_n_dist.extend(sorted(current_dist)[1:n_neighbors+1]) #after distances calculated for current_coord, only select n to save. 
        
    top_n_dist_df = pd.DataFrame(top_n_dist)
    return top_n_dist_df
    



def plot_distance_histogram(top_n_dist_df, n_neighbors = None):
    """
    Plots histogram of euclidean distance between neighbors. Overlays Mean, Median, Std on plot. 

    Parameters
    ----------
    top_n_dist_df : DataFrame
        Saved Euclidean distance values for n_neighbors per coordinate. 
    n_neighbors : int
        Optionally includes # of neighbors in title of plot. 

    Returns
    -------
    None.

    """
     
   
    sns.distplot(top_n_dist_df, norm_hist = True)
    
    plt.title("# Neighbors = " + str(n_neighbors))
    plt.xlim(0,100)
    plt.ylim(0, 0.2)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    
    plt.text(60, 0.18, "Mean = " + str(round(float(np.mean(top_n_dist_df)), 2)))
    plt.text(60, 0.16, "Median = " + str(round(float(np.median(top_n_dist_df)), 2)))
    plt.text(60, 0.14, "Stdev = " + str(round(float(np.std(top_n_dist_df)), 2)))
    plt.show()
    
        
             



#Load Image.
filename_load = '200nm Au NP 20x 0.50 NA collecting obj - 2.2 Cropped; larger.tif' 
file_extension = "C:/Users/Peter/Desktop/Robin's Images/"
filepath = file_extension + filename_load 

image = cv2.imread(filepath, 0) 

laplace_coords = extract_LOG_coordinates(image) #Extract Coordinates from where NP's are. 
plot_overlaid_coordinates(laplace_coords, image) #Plot Identified Coordinates



#Plot Histogram of Euclidean_Distance for each n_neighbors. 
mean_n = []
median_n = []
std_n = []
for n_neighbors in range(1,21): #cycle over range of n_neighbors values. #1-20. 
    top_n_dist_df = neighboring_distance(laplace_coords, n_neighbors, max_distance = 20) 
    plot_distance_histogram(top_n_dist_df, n_neighbors) #plot histogram of Euclidean_Distance for each n_neighbors. 
    
    #Compare summary statistics for each n_neighbors value. 
    mean_n.append(round(float(np.mean(top_n_dist_df)), 2))
    median_n.append(round(float(np.median(top_n_dist_df)), 2))
    std_n.append(round(float(np.std(top_n_dist_df)), 2))
 
    