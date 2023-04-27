# LogisticReg_RandomForest_MLP_TelemarketingDataset
Logistic Regression, Random Forest and Neural Network models are build to decide whether the client subscribed a term deposit or not ? 
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

The dataset can also be obtained from https://archive.ics.uci.edu/ml/datasets/bank+marketing .




### Logistic Regression

Logistic regression is a statistical method used to model the probability of a binary outcome based on one or more predictor variables. 

In logistic regression, the probability of the binary outcome is modeled as a function of one or more predictor variables. The model uses a logistic function (also called the sigmoid function) to transform the linear combination of the predictor variables into a probability value between 0 and 1.

The logistic regression model estimates the coefficients of the predictor variables using maximum likelihood estimation. These coefficients indicate the direction and strength of the relationship between the predictor variables and the binary outcome.

##### Advantage
It can handle both categorical and continuous predictor variables, and it can model nonlinear relationships between the predictor variables and the outcome. 

##### Disadvantage
It assumes that the relationship between the predictor variables and the binary outcome is linear on the logit scale. It can also be sensitive to outliers and the presence of multicollinearity among the predictor variables.


After 5-fold cross validation with 5 repetirions, the best C parameter which gives the highest AUC score

    Best C value: 4.281332398719396



    
![png](output_7_1.png)
    


# Random Forest Model


Random forest classifier is a popular machine learning algorithm used for classification tasks. It is an ensemble learning method that combines multiple decision trees to produce a more accurate and robust model.

The random forest classifier works by building a large number of decision trees, each of which is trained on a random subset of the training data and a random subset of the predictor variables. During training, the algorithm splits the data into subsets based on the values of the predictor variables, and each tree is built using a different subset of the data and predictor variables.

##### Advantage
The final prediction of the random forest classifier is obtained by taking a majority vote of the predictions of all the individual decision trees. This approach helps to reduce overfitting of the model to the training data and increase the robustness of the model to noise in the data.

##### Disadvantage
It can be difficult to interpret the final model, since it is made up of many individual decision trees. However, various methods exist to extract feature importance rankings and to visualize the decision-making process of the individual trees.



After using grid search with 3-fold cross validation with 3 repetetion, the best hyperparameter and the best score is

    Best hyperparameters:  {'max_depth': 50, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 1000}
    Best score:  0.944422141528528


### Neural Network


In a classification problem, a neural network takes a set of input features and produces a probability distribution over the possible classes. During training, the network adjusts its weights and biases to minimize the difference between its predicted probabilities and the true class labels in the training data.

A Multilayer Perceptron (MLP) classifier is a type of neural network used for classification tasks. It is a feedforward neural network that consists of multiple layers of interconnected nodes (neurons).

In an MLP classifier, the input layer consists of the input features, and the output layer consists of the output classes (i.e., the possible categories or labels). The hidden layers in between are composed of a number of neurons, each of which computes a weighted sum of the inputs and applies an activation function to produce an output.

During training, the MLP classifier adjusts its weights and biases to minimize the difference between its predicted output and the true class labels in the training data. This is typically done using an optimization algorithm such as stochastic gradient descent.

##### Advantages
They can learn complex nonlinear relationships between the input features and the output classes, and they can handle both continuous and categorical input features. They are also capable of automatically extracting useful features from the input data, which can reduce the need for manual feature engineering.

##### Disadvantages
MLP classifiers can be computationally intensive and require large amounts of data for training. They can also be prone to overfitting if the model is too complex relative to the amount of training data. Regularization techniques such as dropout and weight decay can help to prevent overfitting and improve the generalization performance of the model.



After using grid search with 3-fold cross validation with 3 repetetion, the best hyperparameter and the best score is

    Best hyperparameters:  {'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10, 10, 10)}
    Best score:  0.93939195041874



### Classification Report

Here 5-fold  cross validation with 5-repetetion technique are not used


    Logistic Regression:
                  precision    recall  f1-score   support
    
               0       0.93      0.97      0.95     36548
               1       0.67      0.42      0.52      4640
    
        accuracy                           0.91     41188
       macro avg       0.80      0.70      0.74     41188
    weighted avg       0.90      0.91      0.90     41188
    
    Random Forest:
                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.98     36548
               1       0.93      0.70      0.80      4640
    
        accuracy                           0.96     41188
       macro avg       0.94      0.85      0.89     41188
    weighted avg       0.96      0.96      0.96     41188
    
    Neural Network:
                  precision    recall  f1-score   support
    
               0       0.95      0.96      0.96     36548
               1       0.66      0.61      0.63      4640
    
        accuracy                           0.92     41188
       macro avg       0.81      0.78      0.80     41188
    weighted avg       0.92      0.92      0.92     41188
    


### COMPARISON
The results show that Random Forest has the highest accuracy and F1-score for predicting the binary classification. Its accuracy is 0.96, and the F1-score is 0.80 for class 1, which means it can predict the minority class with good precision and recall.

Logistic Regression is the second-best model, with an accuracy of 0.91 and an F1-score of 0.52 for class 1. However, it has a lower recall value for class 1, indicating that it may miss some positive cases.

The Neural Network model has an accuracy of 0.92, which is slightly lower than the other two models. Its F1-score for class 1 is 0.63, indicating that it has a decent precision but lower recall for the minority class.

Overall, the Random Forest model seems to be the best model for this binary classification task, followed by Logistic Regression, while Neural Network comes in third.


