# Calculation-of-Particle-Spacing-from-an-Image

Here I calculate the distribution of interparticle spacing between thousands of nanoparticles randomly positioned on a substrate.
<br>
<br>

**Motivation:**
<br>
<br>
My doctoral lab is a leader in the field of making nanoelectrode arrays. We are working on a new design that is dependent on single nanoparticles acting as electrodes, and are fabricated by supporting randomly dispersed particles in a polymer membrane. Unlike our previous designs, the electrode spacing is not well-defined (and effectively random!). Therefore, we needed a method to determine the average electrode spacing - which defines the spatial resolution of our future measurements. 
<br>
<br>

**Approach**
<br>
I tackled this problem by first extracting the (x,y) coordinates from a darkfield image of the nanoparticles. I chose to use a method known as 'Laplacian of the Gaussian' (LOG), which is a spatial filter that finds the edges of features in the image. (It also perfomed the best, of many methods tried). 

<br>

![image](https://user-images.githubusercontent.com/69371709/109753902-e65c0300-7b97-11eb-8fdc-46d1d98265f4.png)


<br>

The yellow circles show the identified particles. The large circles demonstrate that closely packed particles could not be resolved using this method, and are identified as a single unit. Considering that there are <20 large circles, and over 1000 particles total, we concluded that this was acceptable error. 

<br>

Next, we wanted to find the interparticle spacing, though importantly by only considering particles that were direct neighbors. Programmatically identifying direct neighbors was diffiult because it's not only dependent on distance! In fact, if we only considered the nearest 'n' neighbors, there were many instances where the identified closest particles were 'behind' the true neighbors, while igoring particles further away in a direct line of sight. 
<br>
<br>
Luckily, a method exists called "Delaunay Tessellation", which draws trianlges between all points, while maximizing the vertex angles. Interestingly, it is the same algorithm used in finite element simulation to create a 'mesh' pattern - used to define spatial points in physics simulations.
<br>
<br>
The image below show the result of the tessellation on all points in the image. 

![image](https://user-images.githubusercontent.com/69371709/109754018-2327fa00-7b98-11eb-8149-43ebc3d64b24.png)

  
The SciPy implementation outputs all connected vertices, and with a little logic, can easily identify the points that are direct neighbors. See below for two examples, where the red dot is the point in question, and bright yellow spots are identified as direct neighbors. 

<br>

![image](https://user-images.githubusercontent.com/69371709/109754323-a3e6f600-7b98-11eb-840e-0cc096fa9043.png)
![image](https://user-images.githubusercontent.com/69371709/109755546-148f1200-7b9b-11eb-87cb-c3eed3617620.png)


  
 <br>
 
 Lastly, we calculate the Euclidean distance between all neighboring points, where only the distances from the yellow to red dots are considered. The results are displayed in a histogram. 
 
 <br>


![image](https://user-images.githubusercontent.com/69371709/109754220-7dc15600-7b98-11eb-9cd4-4357d06a9aae.png)

  
Overall, this tool serves as a quick and reliable method to characterize the particle spacing in our lab's newly developed arrays! 





<br>
<br>
**Please click each Figure to view in higher resolution!**

<br>

**For a walkthrough of the code, please view the attached Jupyter Notebook (.ipynb)**

