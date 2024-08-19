# credit-risk-classification
Module20
## Overview of the Analysis
- The purpose of this analysis is to use techniques to train and evaluate a model based on loan risk. We want the most accurate model we can get in order to make decisions about future borrowers.    

## What were the variables?
- Financial data points(variable inputs) evaluated in the credit risk model were loan size, interest rate, borrow income, debt-to-income ratio, number of accounts, number of derogatory marks, and the total debt.
- The variables we were trying to describe were low- and high- risk loans.  We have over 18 thousand instances of low-risk loans, and only just over 600 instances of low risk loans.  This shows that the dataset is imbalanced and could skew to favor of making false-positive low-risk loans in a way that would not necessarily be caught by the model.  

## What is the process of ML we went through?
- Put simply, the ML process splits data into testing and training data sets, trains on the the training set and then tests on the testing set. Testing evaluates performance so that metrics can be reevaluated and the model optimized. 
1. We first read the data into a dataframe, segregating the x and y variables into their own sets using pandas techniques.    2. Then we split the data into training and testing data sets using train_test_split from sklearn.
3. Then we create a logistics regression model using sklearn LogisticRegression, and fit this model to our training data sets x_train, y_train split out from train_test_split.
4. Next, test the trained model with our already-split testing data using the predict() function.
5. That Predicted outcome on the testing data is put into a dataframe to review.
6. In the final stages, we evaluate model performance by generating a confusion matrix and then a classification report. 

## Results

### Machine Learning Model 1 (logistic regression):
- Accuracy - 
    - .99 
- Precision - 
    - 1 on low-risk loans 
    - .85 on high-risk loans
- Recall - 
    - .99 on low-risk loans
    - .91 on high-risk loans

### Machine Learning Model 2 (logistic regression with SMOTE to resampling):
- Accuracy - 
    - High- .99 
- Precision - 
    - 1 on low-risk loans 
    - .85 on high-risk loans
- Recall - 
    - .99 on low-risk loans
    - 1 on high-risk loans

### Machine Learning Model 3 (logistic regression with gridsearch hyperparameter tuning):
- Accuracy - 
    - High- .99 
- Precision - 
    - 1 on low-risk loans 
    - .85 on high-risk loans
- Recall - 
    - .99 on low-risk loans
    - 1 on high-risk loans

## Summary

## Which one seems to perform best? How do you know it performs best?
Resampling /balancing this dataset does allow the model to perform better.  Therefore SMOTE resampling would be recommended.  Essentially we have a lot of low-risk loans and not as many high-risk loans. This leads to each high-risk loan having a relatively high weight in the sampling set, underfitting the dataset.  

 Performance does depend on the problem we are trying to solve.  In this instance, by forcing this model to perform better by resampling,  we get more false positive identification of "true low-risk loans."  There are more loans that could default identified and positive low-risk.   This leaves the loan-issuer at higher risk of providing capital to applicants that otherwise would be rejected. 

 It might be good to have a "hybrid" style model, where this more accurate system is used for loans under $30k and then any loan over that amount would be skewed in favor of rejection, thus limiting the loan-issuer's exposure. 
