# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
    The purpose of this analysis is to use techniques to train and evaluate a model based on loan risk. We want the most accurate model we can get in order to make decisions about future borrowers.    
* Explain what financial information the data was on, and what you needed to predict.
    Financial data points(variable inputs) evaluatted in the credit risk model were loan size, interest rate, borrow income, debt-to-income ratio, number of accounts, number of derogatory marks, and the total debt. 
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
    The variables we were trying to describe were low- and high- risk loans.  We have over 18 thousand instances of low-risk loans, and only just over 600 instances of low risk loans.  This shows that the dataset is imbalanced and could skew to favor of making false-positive low-risk loans in a way that would not necessarily be caught by the model.  
* Describe the stages of the machine learning process you went through as part of this analysis.
    Put simply, the ML process splits data into testing and training data sets, trains on the the training set and then tests on the testing set. Testing evaluates performance so that metrics can be reevaluated and the model optimized. 
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
- We first read the data into a dataframe, segregating the x and y variables into their own sets using pandas techniques.    
- Then we split the data into training and testing data sets using train_test_split from sklearn.  
- Then we create a logistics regression model using sklearn LogisticRegression, and fit this model to our training data sets x_train, y_train split out from train_test_split. 
- Next, test the trained model with our already-split testing data using the predict() function.  
- That Predicted outcome on the testing data is put into a dataframe to review. 
- In the final stages, we evaluate model performance by generating a confusion matrix and then a classification report. 

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
