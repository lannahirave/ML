# Homework 2

You are given a Python file with either blank spaces that need to be filled in or bullet points outlining what needs to be implemented. Feel free to modify it as you wish.

1. [x] Load the Flickr30k dataset.
2. [x] Vectorize the images using a method of your choice (preferably the CLIP neural network).
3. [ ] Dimensionality reduction:

   - [x] Implement a PCA class with `fit` and `transform` methods.
   - [x] Visualize PCA with 2 and 3 components using Plotly (in interactive mode).
   - [x] Visualize t-SNE (from sklearn) with 2 and 3 components using Plotly.
   - [ ] Analyze what you observed in 2-4 sentences.
         t-SNE better than PCA IMHO
         t-SNA groups more tightly than pca and looks more neat

4. [x] Clustering:

   - [x] Implement a K-means clustering class.
   - [x] Perform K-means clustering on the original vectors and after PCA with 3 components.
   - [x] Visualize, side-by-side, sample vectors with color-coded cluster labels for both the original vectors and those after PCA with 3 components.
   - [x] Select the optimal number of clusters using a method of your choice.
   - [x] Compare the results and determine which approach is better.
   - [x] Perform hierarchical clustering on the samples after PCA and visualize the results.
   - [x] Compare the hierarchical clustering results with the K-means clusters.
         played mith both methods a litle: k-means depends heavily on input

5. [ ] Outlier detection:

   - [x] Use a sample training split from Homework 1 and apply the DBSCAN algorithm to detect outliers.
   - [x] Visualize the cluster labels of the samples.
   - [x] Remove outliers from the training dataset and retrain your models.

6. [ ] Contrastive search:

   - [x] Vectorize a few textual descriptions.
   - [x] Apply the same dimensionality reduction techniques used for the image vectors (PCA, t-SNE).
   - [x] Search for the nearest 5-10 vectors in the space.
   - [x] Analyze the nearest images in the space relative to a given text input and compare them with the ground truth image descriptions.
   - [x] Save a figure of a few text requests, along with the nearest neighbor results and corresponding descriptions.
