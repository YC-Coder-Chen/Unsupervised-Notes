Unsupervised Learning Notes
============
My study notes, contains Math behind some unsupervised machine learning models.  

K-Means Clustering
------------
> One Sentence Summary:   
Partition observations into a predefined number of homogeneous subgroups by repeatedly updating the centroids of homogeneous subgroups and re-assigning subgroup labels for each data point.  

- **a. Some math notation**  
Suppose we have n samples ![img](https://latex.codecogs.com/svg.latex?X%20%3D%20%5C%7Bx_1%2C%20x_2%2C%20x_3%2C%20...%2C%20x_n%5C%7D), each sample has M features ![img](https://latex.codecogs.com/svg.latex?x_i%20%3D%20%28x_%7B1i%7D%2C%20x_%7B2i%7D%2C%20x_%7B3i%7D%2C%20...%20%2C%20x_%7Bmi%7D%2C%20...%2C%20x_%7BMi%7D%29) and we want to partition them into K clusters ![img](https://latex.codecogs.com/svg.latex?C%20%3D%20%5C%7BC_1%2C%20C_2%2C%20C_3%2C%20...%2C%20C_K%5C%7D). Then all the below formulas should hold true:  

  ![img](https://latex.codecogs.com/svg.latex?C_i%20%5Ccap%20C_j%20%3D%20%5Co%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cbigcup_%7Bi%3D1%7D%5E%7BK%7D%20C_i%20%3D%20X)  

  The above formulas make sure that every sample only belongs to one group. 

- **b. The objective function**  
Firstly, we need to define the distance function to measure the closeness between samples. There are many possible distance functions such as Manhattan distance, Chebyshev distance. Here we use squared Euclidean distance defined below:  

  ![img](https://latex.codecogs.com/svg.latex?d%28x_i%2C%20x_j%29%20%3D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%28x_%7Bim%7D%20-%20x_%7Bjm%7D%29%5E2%20%3D%20%5Cleft%20%5C%7C%20x_i%20-%20x_j%20%5Cright%20%5C%7C)  

  Then based on the above distance function, we define the within cluster variation (the definition here is slightly different from other books and blogs for simplification) to measure the amount by which the observations within a cluster differ from the cluster centroid as below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26W%28C_k%29%20%3D%20%5Cfrac%7B1%7D%7B%7CC_k%7C%7D%20%5Csum_%7Bx_i%20%5Cin%20C_k%7D%20%5Cleft%20%5C%7C%20x_i%20-%20%5Cbar%7Bx_k%7D%20%5Cright%20%5C%7C%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7B%7CC_k%7C%7D%20%5Csum_%7Bx_i%20%5Cin%20C_k%7D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%28x_%7Bim%7D%20-%20%5Cbar%7Bx_%7Bkm%7D%7D%29%5E2%5C%5C%20%5Cend%7Balign*%7D)  

  Where ![img](https://latex.codecogs.com/svg.latex?%7CC_k%7C) is the number of samples inside cluster k and ![img](https://latex.codecogs.com/svg.latex?%5Cbar%7Bx_k%7D) is the centroid of cluster k define below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cbar%7Bx_k%7D%20%3D%20%28%5Cbar%7Bx_%7Bk1%7D%7D%2C%20%5Cbar%7Bx_%7Bk2%7D%7D%2C%20...%20%2C%5Cbar%7Bx_%7Bkm%7D%7D%2C..%2C%20%5Cbar%7Bx_%7BkM%7D%7D%29%5C%5C%20%26%20%5Cboldsymbol%7B%5Ctextsl%7B%20where%20%7D%7D%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cbar%7Bx_%7Bkm%7D%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bx_i%5Cin%20C_k%7D%20x_%7Bim%7D%7D%7B%7CC_k%7C%7D%5C%5C%20%5Cend%7Balign*%7D)  

  Intuitively, a perfect clustering is the one for which the within cluster variation is as small as possible. So our objective is to minimize the total distance between samples and corresponding centroids define as below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26C%5E*%20%3D%20arg%5Cunderset%7BC%7D%7BMin%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20W%28C_k%29%20%5C%5C%20%26%20%3D%20arg%5Cunderset%7BC%7D%7BMin%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%28%5Csum_%7Bx_i%20%5Cin%20C_k%7D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%28x_%7Bim%7D%20-%20%5Cbar%7Bx_%7Bkm%7D%7D%29%5E2%29%20%5Cend%7Balign*%7D)  

- **c. More realistic approach**  
The Loss function defined above seems perfect. But in reality, it is an NP-hard problem, which means that it is so hard to figure out the global optimal solution. Specifically, we need to try ![img](https://latex.codecogs.com/svg.latex?S%28n%2C%20K%29) times:

  ![img](https://latex.codecogs.com/svg.latex?S%28n%2C%20K%29%20%3D%20%5Cfrac%7B1%7D%7BK%21%7D%5Csum_%7Bl%3D1%7D%5E%7BK%7D%20%28-1%29%5E%7BK-l%7D%5Cbinom%7BK%7D%7Bl%7DK%5En)  

  Instead of spending so much computing power on finding the global optimal, we search for the local optimal instead.  

- **d. K-means clustering algorithm**  
*Model Input*:  n samples, ![img](https://latex.codecogs.com/svg.latex?X%20%3D%20%5C%7Bx_1%2C%20x_2%2C%20x_3%2C%20...%2C%20x_n%5C%7D)   
*Model Output*: Final clusters C, where ![img](https://latex.codecogs.com/svg.latex?C%20%3D%20%5C%7BC_1%2C%20C_2%2C%20C_3%2C%20...%2C%20C_K%5C%7D)  

  *Steps*:  
  
  (1) Initialization:  
  - Set t = 0
  - Ramdomly select K samples as the cluster centroids ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%280%29%7D) as below: 
  
    ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%280%29%7D%20%3D%20%5C%7BM_1%5E%7B%280%29%7D%2C%20M_2%5E%7B%280%29%7D%2C%20...%2C%20M_k%5E%7B%280%29%7D%2C%20...%2C%20M_K%5E%7B%280%29%7D%5C%7D)  

  (2) Cluster each samples:  
  - For cluster centroids ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%28t%29%7D), where 

    ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%28t%29%7D%20%3D%20%5C%7BM_1%5E%7B%28t%29%7D%2C%20M_2%5E%7B%28t%29%7D%2C%20...%2C%20M_k%5E%7B%28t%29%7D%2C%20...%2C%20M_K%5E%7B%28t%29%7D%5C%7D) and ![img](https://latex.codecogs.com/svg.latex?M_k%5E%7B%28t%29%7D) is the centroid of cluster k, 

    We compute the distance between each samples and each centroids, and then assign each sample to the cluster that have the closest centroid. Now we will have new cluster result ![img](https://latex.codecogs.com/svg.latex?C%5E%7B%28t%29%7D), and the cluster result for each sample ![img](https://latex.codecogs.com/svg.latex?C_%7Bx_i%7D%5E%7B%28t%29*%7D) is as below:  

    ![img](https://latex.codecogs.com/svg.latex?C_%7Bx_i%7D%5E%7B%28t%29*%7D%20%3D%20arg%5Cunderset%7BC%7D%7BMin%7D%20%5Cleft%20%5C%7C%20x_i%20-%20M_C%5E%7B%28t%29%7D%20%5Cright%20%5C%7C)  

  (3) Recompute the centroids:  
  - For the cluster result ![img](https://latex.codecogs.com/svg.latex?C%5E%7B%28t%29%7D) at iteration t, we compute the updated centroids 
  ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%28t&plus;1%29%7D) as below:  

    ![img](https://latex.codecogs.com/svg.latex?M%5E%7B%28t&plus;1%29%7D%20%3D%20%5C%7BM_1%5E%7B%28t&plus;1%29%7D%2C%20M_2%5E%7B%28t&plus;1%29%7D%2C%20...%2C%20M_k%5E%7B%28t&plus;1%29%7D%2C%20...%2C%20M_K%5E%7B%28t&plus;1%29%7D%5C%7D)  

    where   
    ![img](https://latex.codecogs.com/svg.latex?M_k%5E%7B%28t&plus;1%29%7D%20%3D%20%28%20%5Cfrac%7B%5Csum_%7Bx_i%5Cin%20%7BC_k%5E%7B%28t&plus;1%29%7D%7D%7D%20x_%7Bi1%7D%7D%7B%7CC_k%5E%7B%28t&plus;1%29%7D%7C%7D%2C%20%5Cfrac%7B%5Csum_%7Bx_i%5Cin%20%7BC_k%5E%7B%28t&plus;1%29%7D%7D%7D%20x_%7Bi2%7D%7D%7B%7CC_k%5E%7B%28t&plus;1%29%7D%7C%7D%2C%20...%2C%5Cfrac%7B%5Csum_%7Bx_i%5Cin%20%7BC_k%5E%7B%28t&plus;1%29%7D%7D%7D%20x_%7Bim%7D%7D%7B%7CC_k%5E%7B%28t&plus;1%29%7D%7C%7D%2C%20%5Cfrac%7B%5Csum_%7Bx_i%5Cin%20%7BC_k%5E%7B%28t&plus;1%29%7D%7D%7D%20x_%7BiM%7D%7D%7B%7CC_k%5E%7B%28t&plus;1%29%7D%7C%7D%29)  
  
  (4) Check iteration status:
  - If ![img](https://latex.codecogs.com/svg.latex?C%5E%7B%28t%29%7D) = ![img](https://latex.codecogs.com/svg.latex?C%5E%7B%28t-1%29%7D) for a few iteration, then we stop and output ![img](https://latex.codecogs.com/svg.latex?C%5E*%20%3D%20C%5E%7B%28t%29%7D)  
  - Else, set t = t+1 and go back to step (2)  

**Reference**  

1. Dubes R C, Jain A K. Algorithms for clustering data[J]. 1988.  
2. Gan G, Ma C, Wu J. Data clustering: theory, algorithms, and applications[M]. Siam, 2007.  
3. Hang Li. Statistical Learning Method[M]. Tsinghua University Press, 2019. [Chinese]
4. https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1