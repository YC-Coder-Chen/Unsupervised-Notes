Unsupervised Learning Notes
============
Machine Learning study notes, contains Math behind some unsupervised machine learning models.  

Principal Components Analysis
------------
> One Sentence Summary:   
Use a few unrelated features to represent original features in the dataset and retain as much information (variance) as possible.  

- **a. Some math notation**  
Suppose our dataset has m dimensions ![img](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%3D%20%28x_1%2C%20x_2%2C%20...%2C%20x_m%29%5ET), and its Mean Vector is ![img](https://latex.codecogs.com/svg.latex?%5Cmathbf%7B%5Cmu%7D%20%3D%20E%28x%29%20%3D%20%28%5Cmu_1%2C%20%5Cmu_2%2C%20...%2C%20%5Cmu_m%29%5ET), its Covariance Matrix is ![img](https://latex.codecogs.com/svg.latex?%5Cmathbf%7B%5CSigma%7D%20%3D%20Cov%28x%2C%20x%29%20%3D%20E%5B%28x-%5Cmu%29%28x-%5Cmu%29%5ET%5D).  

  Now we consider the below linear transformation from m dimensional random variable x to m dimension random variable y with ![img](https://latex.codecogs.com/svg.latex?%5Cmathbf%7By%7D%20%3D%28y_1%2C%20y_2%2C%20...%2C%20y_m%29%5ET): 

  ![img](https://latex.codecogs.com/svg.latex?y_i%20%3D%20%5Calpha_i%5ET%5Cmathbf%7Bx%7D%20%3D%20%5Calpha_%7B1i%7Dx_1%20&plus;%20%5Calpha_%7B2i%7Dx_2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmi%7Dx_m)  

  where ![img](https://latex.codecogs.com/svg.latex?%5Calpha_i%5ET%20%3D%20%28%5Calpha_%7B1i%7D%2C%20%5Calpha_%7B2i%7D%2C%20...%2C%20%5Calpha_%7Bmi%7D%29%2C%20i%20%3D%201%2C2%2C3%2C...%2Cm).  

- **b. Definition of Principal Components**  
  Suppose we have the same linear transformation setting as above:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cmathbf%7By%7D%20%3D%28y_1%2C%20y_2%2C%20...%2C%20y_m%29%5ET)  

  with ![img](https://latex.codecogs.com/svg.latex?y_i%20%3D%20%5Calpha_i%5ET%5Cmathbf%7Bx%7D%20%3D%20%5Calpha_%7B1i%7Dx_1%20&plus;%20%5Calpha_%7B2i%7Dx_2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmi%7Dx_m)  

  where ![img](https://latex.codecogs.com/svg.latex?%5Calpha_i%5ET%20%3D%20%28%5Calpha_%7B1i%7D%2C%20%5Calpha_%7B2i%7D%2C%20...%2C%20%5Calpha_%7Bmi%7D%29%2C%20i%20%3D%201%2C2%2C3%2C...%2Cm)  

  Then we call ![img](https://latex.codecogs.com/svg.latex?y_1%2C%20y_2%2C%20...%2C%20y_m) as the first principal component, second principal component, ..., the mth principal component correspondingly if they meets the below requirements:  

  - (1) Coefficient vector ![img](https://latex.codecogs.com/svg.latex?%5Calpha_i%5ET) is a unit vector with:  

    ![img](https://latex.codecogs.com/svg.latex?%5Calpha_i%5ET%5Calpha_i%20%3D%20%5Calpha_%7B1i%7D%5E2%20&plus;%20%5Calpha_%7B2i%7D%5E2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmi%7D%5E2%20%3D%201)  
  
  - (2) Variable ![img](https://latex.codecogs.com/svg.latex?y_i) and ![img](https://latex.codecogs.com/svg.latex?y_j) are uncorrelated with:    

     ![img](https://latex.codecogs.com/svg.latex?Cov%28y_i%2C%20y_j%29%20%3D%200%2C%20%5C%2C%20%5C%2C%20%5C%2C%20when%5C%2C%20%5C%2C%20%5C%2C%20i%5Cneq%20j)  

  - (3) Variable ![img](https://latex.codecogs.com/svg.latex?y_1) has the highest variance among all possbile linear transformations of x. Variable ![img](https://latex.codecogs.com/svg.latex?y_2) has the highest variance among all possbile linear transformations of x that are uncorrelated with ![img](https://latex.codecogs.com/svg.latex?y_1). Generally, ![img](https://latex.codecogs.com/svg.latex?y_i) has the highest variance among all possbile linear transformations of x that are uncorrelated with ![img](https://latex.codecogs.com/svg.latex?y_1%2C%20y_2%2C%20...%2C%20y_%7Bi-1%7D). 
  
  Specifically, we are trying to solve the below optimization problem:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26y_i%20%3D%20arg%5Cunderset%7By_i%7D%7BMax%7D%20%5C%2C%20%5C%2C%20%5C%2C%20Var%28y_i%29%20%3D%20%5Calpha_i%5ET%5CSigma%20%5Calpha_i%5C%5C%20%26s.t.%20%5C%2C%20%5C%2C%20%5C%2C%20Cov%28y_i%2C%20y_j%29%20%3D%200%2C%20%5C%2C%20%5C%2C%20%5C%2C%20for%20%5C%2C%20%5C%2C%20%5C%2C%20j%20%3D%201%2C2%2C3%2C...%2Ci-1%5C%5C%20%26s.t.%20%5C%2C%20%5C%2C%20%5C%2C%20%5Calpha_i%5ET%5Calpha_i%20%3D%201%20%5Cend%7Balign*%7D)  
  
- **c. Eigenvalues, Eigenvectors & Principal Components**  
  - What is Eigenvalues & Eigenvectors ?  

    For a matrix A with dimension (mxm), if the below function holds true and vector v is not zero-vector:  

    ![img](https://latex.codecogs.com/svg.latex?Av%3D%20%5Clambda%20v)


    Then vector v is the Eigenvector and ![img](https://latex.codecogs.com/svg.latex?%5Clambda) is the Eigenvalue.  

  - The relationship between Eigenvalues, Eigenvectors & Principal Components  

    Suppose ![img](https://latex.codecogs.com/svg.latex?%5CSigma) is the covariance matrix with dimension (mxm), the Eigenvalues of ![img](https://latex.codecogs.com/svg.latex?%5CSigma) is ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1%2C%20%5Clambda_2%2C%20...%2C%20%5Clambda_m) with ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1%20%5Cgeqslant%20%5Clambda_2%20%5Cgeqslant%20...%20%5Cgeqslant%20%5Clambda_m%20%5Cgeqslant%200), and the corresponding unit Engenvectors of the Engenvalues are ![img](https://latex.codecogs.com/svg.latex?%5Calpha_1%2C%20%5Calpha_2%2C%20...%2C%20%5Calpha_m),  

    then we can prove that the kth principal component of x is:  

    ![img](https://latex.codecogs.com/svg.latex?y_k%20%3D%20%5Calpha_k%5ETx%20%3D%20%5Calpha_%7B1k%7Dx_1%20&plus;%20%5Calpha_%7B2k%7Dx_2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmk%7Dx_m)  

    and the variance of kth principal component is:  

    ![img](https://latex.codecogs.com/svg.latex?Var%28y_k%29%20%3D%20%5Calpha_k%5ET%5CSigma%20%5Calpha_k%20%3D%20%5Clambda_k%2C%20%5C%2C%20%5C%2C%20%5C%2C%20k%20%3D%201%2C2%2C...%2Cm)  

- **d. Proof on the relationship between Eigenvalues, Eigenvectors & Principal Components**  
  - For the first principal components, our target is:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7B%5Calpha_1%7D%7BMax%7D%20%5C%2C%20%5C%2C%20Var%28y_1%29%20%3D%20Var%28%5Calpha_1%5ET%5CSigma%20%5Calpha_1%29%20%5C%5C%20%26s.t.%20%5C%2C%20%5C%2C%20%5Calpha_1%5ET%5Calpha_1%20%3D%201%20%5Cend%7Balign*%7D)  

    We then define the lagrange function as below:  

    ![img](https://latex.codecogs.com/svg.latex?L%28%5Calpha_1%29%20%3D%20%5Calpha_1%5ET%5CSigma%20%5Calpha_1%20-%20%5Clambda_1%28%20%5Calpha_1%5ET%20%5Calpha_1-1%29)  

    At optimal point, the above function should satify the Stationarity condition in K.K.T. condition as below:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20L%28%5Calpha_1%29%7D%7B%5Cpartial%20%7B%5Calpha_1%7D%7D%20%3D%202%28%5CSigma%5Calpha_1%20-%20%5Clambda_1%5Calpha_1%29%20%3D%200)  

    So based on the definition of Eigenvector and Eigenvalue, we can easily figure out that ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1) is the Eigenvalue and ![img](https://latex.codecogs.com/svg.latex?%5Calpha_1) is the unit Eigenvector because ![img](https://latex.codecogs.com/svg.latex?%5Calpha_1%5ET%5Calpha_1%20%3D%201).  

    And so ![img](https://latex.codecogs.com/svg.latex?Var%28y_1%29%20%3D%20Var%28%5Calpha_1%5ETx%29%20%3D%20%5Calpha_1%5ET%5CSigma%5Calpha_1%20%3D%20%5Calpha_1%5ET%5Clambda_1%20%5Calpha_1%20%3D%20%5Clambda_1%20%5Calpha_1%5ET%20%5Calpha_1%20%3D%20%5Clambda_1).  


  - For the second principal components, out target is:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7B%5Calpha_2%7D%7BMax%7D%20%5C%2C%20%5C%2C%20Var%28y_2%29%20%3D%20Var%28%5Calpha_2%5ET%5CSigma%20%5Calpha_2%29%20%5C%5C%20%26s.t.%20%5C%2C%20%5C%2C%20%5Calpha_2%5ET%5Calpha_2%20%3D%201%5C%5C%20%26%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Calpha_1%5ET%5CSigma%20%5Calpha_2%20%3D%200%5C%5C%20%5Cend%7Balign*%7D)  

    Similarly, we then define the lagrange function as below:  

    ![img](https://latex.codecogs.com/svg.latex?L%28%5Calpha_2%29%20%3D%20%5Calpha_2%5ET%5CSigma%20%5Calpha_2%20-%20%5Clambda_2%28%20%5Calpha_2%5ET%20%5Calpha_2-1%29%20-%20%5Cphi%5Calpha_1%5ET%5CSigma%20%5Calpha_2)  

    Notice that:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cphi%5Calpha_1%5ET%5CSigma%20%5Calpha_2%20%3D%20%5Cphi%5Calpha_2%5ET%5CSigma%20%5Calpha_1%20%3D%20%5Cphi%5Calpha_2%5ET%5Clambda_1%20%5Calpha_1%20%3D%20%5Cphi%5Clambda_1%20%5Calpha_2%5ET%20%5Calpha_1%20%3D%20%5Cbar%7B%5Cphi%20%7D%5Calpha_2%5ET%20%5Calpha_1)  

    So at optimal point, the above lagrange function should satify the Stationarity condition in K.K.T. condition as below:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cfrac%7B%5Cpartial%20L%28%5Calpha_2%29%7D%7B%5Cpartial%20%7B%5Calpha_2%7D%7D%20%3D%202%5CSigma%5Calpha_2%20-%202%5Clambda_2%5Calpha_2%20-%20%5Cbar%7B%5Cphi%7D%5Calpha_1%20%3D%200%20%5C%5C%20%26%5CRightarrow%202%5Calpha_1%5ET%5CSigma%5Calpha_2%20-%202%5Calpha_1%5ET%5Clambda_2%5Calpha_2%20-%20%5Cbar%7B%5Cphi%7D%5Calpha_1%5ET%5Calpha_1%20%3D%200%5C%5C%20%26%5CRightarrow%20%5Cbar%7B%5Cphi%7D%5Calpha_1%5ET%5Calpha_1%20%3D%200%5C%5C%20%26%5CRightarrow%20%5Cbar%7B%5Cphi%7D%3D%200%5C%5C%20%26%5CRightarrow%202%5CSigma%5Calpha_2%20-%202%5Clambda_2%5Calpha_2%20%3D%200%5C%5C%20%26%5CRightarrow%20%5CSigma%5Calpha_2%20-%20%5Clambda_2%5Calpha_2%20%3D%200%5C%5C%20%5Cend%7Balign*%7D)  

    So based on the definition of Eigenvector and Eigenvalue, we can easily figure out that ![img](https://latex.codecogs.com/svg.latex?%5Clambda_2) is the Eigenvalue and ![img](https://latex.codecogs.com/svg.latex?%5Calpha_2) is the unit Eigenvector because ![img](https://latex.codecogs.com/svg.latex?%5Calpha_2%5ET%5Calpha_2%20%3D%201).  

    And so ![img](https://latex.codecogs.com/svg.latex?Var%28y_2%29%20%3D%20Var%28%5Calpha_2%5ETx%29%20%3D%20%5Calpha_2%5ET%5CSigma%5Calpha_2%20%3D%20%5Calpha_2%5ET%5Clambda_2%20%5Calpha_2%20%3D%20%5Clambda_2%20%5Calpha_2%5ET%20%5Calpha_2%20%3D%20%5Clambda_2). 

  - So similarly, we can prove that the kth principal component of x is:  

    ![img](https://latex.codecogs.com/svg.latex?y_k%20%3D%20%5Calpha_k%5ETx%20%3D%20%5Calpha_%7B1k%7Dx_1%20&plus;%20%5Calpha_%7B2k%7Dx_2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmk%7Dx_m)  

    and the variance of kth principal component is:  

    ![img](https://latex.codecogs.com/svg.latex?Var%28y_k%29%20%3D%20%5Calpha_k%5ET%5CSigma%20%5Calpha_k%20%3D%20%5Clambda_k%2C%20%5C%2C%20%5C%2C%20%5C%2C%20k%20%3D%201%2C2%2C...%2Cm)   

    Where ![img](https://latex.codecogs.com/svg.latex?%5CSigma) is the covariance matrix with dimension (mxm), the Eigenvalues of ![img](https://latex.codecogs.com/svg.latex?%5CSigma) is ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1%2C%20%5Clambda_2%2C%20...%2C%20%5Clambda_m) with ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1%20%5Cgeqslant%20%5Clambda_2%20%5Cgeqslant%20...%20%5Cgeqslant%20%5Clambda_m%20%5Cgeqslant%200), and the corresponding unit Engenvectors of the Engenvalues are ![img](https://latex.codecogs.com/svg.latex?%5Calpha_1%2C%20%5Calpha_2%2C%20...%2C%20%5Calpha_m).  

- **e. How to select the right number of Principal Components**  
  Above we talked about the logic behind figuring out the mth principal components. In fact, among these m principal components, we only need k of them and k principal components already retained much information (variance). Specifically, we use the below Proportion of Variance Explained metric to define the right k:  

  Proportion of Variance Explained(k)= ![img](https://latex.codecogs.com/svg.latex?%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Ceta_i%20%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Clambda_i%20%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Clambda%20_i%20%7D)  

  Where ![img](https://latex.codecogs.com/svg.latex?%5Ceta_i) is the variance of the ith principal component.  

  Usually, we select the k that makes Proportion of Variance Explained greater than 70%-80%, which means that by selecting the top k principal components, we already retained about 70%-80% of the information (variance).  
- **f. Actual Algorithms**  
  There are two main algorithms (Eigenvalues Based Algorithm & SVD Based Algorithm) to decompose the dataset with m features into m principal components. Here we will only introduce the Eigenvalues Based Algorithm.  

  - Eigenvalues Based Algorithm  
    Suppose now we have a dataset X with shape(mxn), m is the number of features and n is the number of samples.  
    - (1) Firstly, normal dataset X. We have to make sure that features with different scales are weighted the same.  

      Specifically, for each ![img](https://latex.codecogs.com/svg.latex?x_%7Bij%7D) in Matrix X, we will have:  

      ![img](https://latex.codecogs.com/svg.latex?x_%7Bij%7D%5E*%20%3D%20%5Cfrac%7Bx_%7Bij%7D%20-%20%5Cbar%7Bx_i%7D%7D%7B%5Csqrt%7Bs_%7Bii%7D%7D%7D)  

      Where  

      ![img](https://latex.codecogs.com/svg.latex?%5Cbar%7Bx_i%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20x_%7Bij%7D%2C%20%5C%2C%20%5C%2C%20%5C%2C%20i%3D1%2C2%2C...%2Cm)  

      ![img](https://latex.codecogs.com/svg.latex?%7Bs_%7Bii%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%28x_%7Bij%7D%20-%20%5Cbar%7Bx_i%7D%29%5E2)  

    - (2) Secondly, compute the covariance matrix R, recall that right now X is the normalied dataset:  

      ![img](https://latex.codecogs.com/svg.latex?R%20%3D%20%5Br_%7Bij%7D%5D_%7Bm*m%7D%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7DXX%5ET)  

      Where  

      ![img](https://latex.codecogs.com/svg.latex?r_%7Bij%7D%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bl%3D1%7D%5E%7Bn%7Dx_%7Bil%7Dx_%7Blj%7D%2C%20%5C%2C%20%5C%2C%20%5C%2C%20i%2Cj%20%3D%201%2C2%2C3%2C...%2Cm)  

    - (3) Then, compute the Eigenvectors and Eigenvalues by solving the characteristic equation below:  

      ![img](https://latex.codecogs.com/svg.latex?%7CR%20-%20%5Clambda%20I%7C%20%3D%200)  

      Now we will have m Eigenvalues ![img](https://latex.codecogs.com/svg.latex?%5Clambda_1%20%5Cgeqslant%20%5Clambda_2%20%5Cgeqslant%20...%20%5Cgeqslant%20%5Clambda_m%20%5Cgeqslant%200) and also m corresponding Eigenvectors ![img](https://latex.codecogs.com/svg.latex?%5Calpha_1%2C%20%5Calpha_2%2C%20...%2C%20%5Calpha_m).  

    - (4) Select the right number of k by figuring out the k that makes the  Proportion of Variance Explained matrix ![img](https://latex.codecogs.com/svg.latex?%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Ceta_i%20%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Clambda_i%20%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Clambda%20_i%20%7D) greater than the predefined threshold. 

    - (5) Compute K principal components:  

      Specifically, the kth principal component of x is:  

      ![img](https://latex.codecogs.com/svg.latex?y_k%20%3D%20%5Calpha_k%5ETx%20%3D%20%5Calpha_%7B1k%7Dx_1%20&plus;%20%5Calpha_%7B2k%7Dx_2%20&plus;%20...%20&plus;%20%5Calpha_%7Bmk%7Dx_m)  

**Reference**  

1. Jolliffe I. Principal component analysis[M]. Springer Berlin Heidelberg, 2011.  
2. Smith L I. A tutorial on principal components analysis[R]. 200
3. Hang Li. Statistical Learning Method[M]. Tsinghua University Press, 2019. [Chinese]  
4. https://en.wikipedia.org/wiki/Principal_component_analysis  
5. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  