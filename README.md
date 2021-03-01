# Calculation-of-Particle-Spacing-from-an-Image
<br>
Here I calculate the distribution of interparticle spacing between thousands of nanoparticles randomly positioned on a substrate. 

<br>

I use a Laplacian of Gaussian (LOG) spatial filter to extract each particle's (x,y) coordinates from the input image. 

<br>

I then calculate the Euclidean distance to each particle's direct neighbors, found using Delaunay Tessellation. The results are summarized in a histogram. 


**For a walkthrough of the code, please view the attached Jupyter Notebook (.ipynb)**

