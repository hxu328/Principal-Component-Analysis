# Principal-Component-Analysis
implementing a facial analysis program using Principal Component Analysis (PCA), using the skills including algebra + PCA


## Dataset
	[The Extended Yale Face Database B](https://www.example.com)
  

## Program Specification

1. load_and_center_dataset(filename) — load the dataset from a provided .npy file, re-center it around the origin and return it as a NumPy array of floats

2. get_covariance(dataset) — calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)

3. get_eig(S, m) — perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns

4. get_eig_perc(S, perc) — similar to get_eig, but instead of returning the first m, return all eigenvalues and corresponding eigenvectors in similar format as get_eig that explain more than perc % of variance

5. project_image(image, U) — project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array

6. display_image(orig, proj) — use matplotlib to display a visual representation of the original image and the projected image side-by-side
