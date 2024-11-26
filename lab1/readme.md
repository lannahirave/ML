You are given two Python files with either blank spaces that need to be filled in or bullet points outlining what needs to be implemented.

## Part 1:

- Implement your own Linear Regression model class:

  Initialize the weights

  Verbose training process: log training statistics (e.g., cost)

  Implement the `fit` method

  Implement the `predict` method

  The plot of your model's performance should closely resemble that of Sklearn. If it does not, try tuning the hyperparameters (don't hesitate to set `num_iterations` to 1,000,000).

  # learning rate from 1e-5 to 1e-4 reduced iterations from 600k to 100k approx

- Write a function for the Normal Equation: estimate the weights without training the model:

  - [x] Done

## Part 2:

You are given a small subset of the Flickr30k dataset.

1. [x] Load the data.
2. [ ] Convert the images of shape (3, H, W) into N-dimensional vectors and explain how you did it. A few ideas include:

   - [ ] Convert to grayscale and flatten.
   - [ ] Take the mean across the RGB channels and flatten.
   - [x] Use any pre-trained embedding model (e.g., CLIP, SigLIP, etc.). [used cliip from openai]
   - [ ] Any other method you find suitable.

3. [x] Convert the text labels to integers: 0 for humans, 1 for animals.
4. [x] Create a train/test split. Use a test size of 0.2 to evaluate your final model.
5. [ ] Train the following models:

   - [x] LogisticRegression (implement)
   - [x] KNN
   - [x] DecisionTree

6. [ ] Train the model using the following validation strategies:

   - [x] **Simple train/test split**: Further divide the training data into training and validation subsets for small-scale hyperparameter optimization based on the validation set's performance.
   - [x] **K-fold validation**: Train a separate model on each fold and evaluate the model using K-fold cross-validation.
   - [x] **Stratified K-fold**: Similar to K-fold validation but ensures that class proportions are maintained across the splits.

7. [x] Make predictions on the test set and measure accuracy.

   - [x] For the simple train/test split, you will have only one model, so no additional modifications are needed.
   - [x] In the K-fold case, you will have K models, so you can experiment with methods like majority voting, absolute voting, or average probability aggregation with a threshold.

8. [ ] Perform error analysis:

   - [ ] Examine the samples where the model predicted the wrong label and provide an explanation for why you think it happened.
   - [x] Visualize the confusion matrix, showing counts of correctly classified classes and misclassified ones.
         Find in part2/img
   - [x] Try to improve your model: data cleaning, hyperparameters, model choice, etc.

   **Model: logistic_regression**
   Simple train/test accuracy: 0.92
   K-Fold Validation (k=5) Accuracy: 0.88
   Stratified K-Fold Validation (k=5) Accuracy: 0.88
   Time: 4.94s

   **Model: knn**
   Simple train/test accuracy: 0.90
   K-Fold Validation (k=5) Accuracy: 0.88
   Stratified K-Fold Validation (k=5) Accuracy: 0.88
   Time: 0.70s

   **Model: decision_tree**
   Simple train/test accuracy: 0.68
   K-Fold Validation (k=5) Accuracy: 0.73
   Stratified K-Fold Validation (k=5) Accuracy: 0.72
   Time: 1.28s

   KNN significantly faster than logistic but 2% worse in simple train/test
