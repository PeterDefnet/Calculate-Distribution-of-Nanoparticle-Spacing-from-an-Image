# Calculate-Nanoparticle-Spacing-from-an-Image
Here I calculate the distribution of interparticle spacing between thousands of nanoparticles randomly positioned on a substrate. I used a Laplacian of Gaussian (LOG) spatial filter to extract each particle's (x,y) coordinates from the input image. I then calculated the Euclidean distance to each particle's direct neighbors, found using Delauney tessellation. 

