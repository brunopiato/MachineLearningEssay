
# Machine Learning Essay

Author: Bruno Piato

Date: 11 September 2023

## About

A systematic essay is the best way to compare different algorithms and hyperparameter fine-tuning in terms of their performance metrics. Here I tested five classification algorithms, eleven regression models and two clustering algorithms.

This essay is the final project of the Machine Learning Fundamentals discipline taken from July to September in Comunidade Data Science. The discipline was an introduction to the main concepts and algorithms used in machine learning and data science projects.

Each one of the algorithms of the same kind was trained using the same dataset so the observed differences are only due to algorithm's properties. The best results for each algorithm hyperparameter fine-tuning is shown in the tables bellow so one can compare them side-by-side in terms of the main performance metrics for that type of algorithms.

As the final product of this project I wrote this report and I also developed and published a WebApp using Streamlit so the user can better visualizer the results found here. It can bem accessed in https://machinelearningevaluator.streamlit.app/.


# Classification Algorithms Results

For the classification kind of algorithms I implemented K Nearest Neighbors, Decision Tree Classifier, Random Forest Classifier and Logistic Regression classifier. These are not necessariyl the recently most used algorithms for classification, but they are the fundamentals from where one can study and understand classification problem solving and predicting.

## Classification Training Results
| Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|:----------|------:|----------------:|----------------:|----------------------:|
| Accuracy  | 0.958 |           0.978 |           0.978 |                 0.875 |
| Precision | 0.974 |           0.985 |           0.985 |                 0.87  |
| Recall    | 0.928 |           0.963 |           0.963 |                 0.836 |
| F1-Score  | 0.951 |           0.974 |           0.974 |                 0.853 |


## Classification Validation Results
| Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|:----------|------:|----------------:|----------------:|----------------------:|
| Accuracy  | 0.925 |           0.95  |           0.957 |                 0.874 |
| Precision | 0.942 |           0.951 |           0.963 |                 0.87  |
| Recall    | 0.881 |           0.933 |           0.936 |                 0.835 |
| F1-Score  | 0.911 |           0.942 |           0.949 |                 0.852 |


## Classification Test Results
| Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|:----------|------:|----------------:|----------------:|----------------------:|
| Accuracy  | 0.925 |           0.951 |           0.957 |                 0.872 |
| Precision | 0.942 |           0.953 |           0.964 |                 0.869 |
| Recall    | 0.883 |           0.933 |           0.936 |                 0.834 |
| F1-Score  | 0.911 |           0.943 |           0.95  |                 0.851 |

---
# Regression Algorithms Results
For the regression kind of algorithms I implemented Linear Models (Lasso, Ridge and ElasticNet), Decision Tree Regressor, Random Forest Regressor, Polinomial Regression (Lasso, Ridge and ElasticNet) and RANSAC. These are not necessarily the recently most used algorithms for regression, but they are the fundamentals from where one can study and understand regression problem solving and predicting.

### Training dataset
|         Model                    |     R2 |      MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|-------:|---------:|-------:|-------:|-------:|
| Linear Model                     |  0.045 |  456.561 | 21.367 | 17.015 |  8.628 |
| Linear Model Lasso               |  0.007 |  474.475 | 21.782 | 17.305 |  8.737 |
| Linear Model Ridge               |  0.045 |  456.561 | 21.367 | 17.015 |  8.628 |
| Linear Model ElasticNet          |  0.008 |  474.269 | 21.778 | 17.3   |  8.732 |
| Decision Tree Regressor          |  0.992 |    3.94  |  1.985 |  0.214 |  0.083 |
| Random Forest Regressor          |  0.903 |   46.163 |  6.794 |  4.871 |  2.581 |
| Polinomial Regression            |  0.087 |  436.62  | 20.895 | 16.571 |  8.357 |
| Polinomial Regression Lasso      |  0.014 |  471.28  | 21.709 | 17.23  |  8.649 |
| Polinomial Regression Ridge      |  0.086 |  437.12  | 20.907 | 16.58  |  8.379 |
| Polinomial Regression ElasticNet |  0.013 |  471.878 | 21.723 | 17.244 |  8.679 |
| RANSAC                           | -1.12  | 1013.6   | 31.837 | 24.762 | 10.508 | 

### Validation Dataset
|         Model                    |     R2 |      MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|-------:|---------:|-------:|-------:|-------:|
| Linear Model                     |  0.04  |  458.197 | 21.406 | 17.041 |  8.663 |
| Linear Model Lasso               |  0.008 |  473.747 | 21.766 | 17.265 |  8.696 |
| Linear Model Ridge               |  0.04  |  458.196 | 21.406 | 17.041 |  8.663 |
| Linear Model ElasticNet          |  0.008 |  473.636 | 21.763 | 17.263 |  8.694 |
| Decision Tree Regressor          | -0.318 |  629.213 | 25.084 | 17.115 |  6.967 |
| Random Forest Regressor          |  0.332 |  318.964 | 17.86  | 13.065 |  7.062 |
| Polinomial Regression            |  0.065 |  446.483 | 21.13  | 16.79  |  8.541 |
| Polinomial Regression Lasso      |  0.014 |  470.756 | 21.697 | 17.181 |  8.656 |
| Polinomial Regression Ridge      |  0.067 |  445.744 | 21.113 | 16.78  |  8.555 |
| Polinomial Regression ElasticNet |  0.013 |  471.408 | 21.712 | 17.2   |  8.675 |
| RANSAC                           | -1.204 | 1052.24  | 32.438 | 25.325 | 10.545 | 

### Testing Dataset
|           Model                  |     R2 |      MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|-------:|---------:|-------:|-------:|-------:|
| Linear Model                     |  0.048 |  463.691 | 21.533 | 17.178 |  8.528 |
| Linear Model Lasso               |  0.008 |  483.178 | 21.981 | 17.473 |  8.753 |
| Linear Model Ridge               |  0.048 |  463.691 | 21.533 | 17.178 |  8.528 |
| Linear Model ElasticNet          |  0.008 |  483.035 | 21.978 | 17.47  |  8.745 |
| Decision Tree Regressor          | -0.252 |  609.704 | 24.692 | 17.156 |  6.267 |
| Random Forest Regressor          |  0.347 |  318.041 | 17.834 | 13.122 |  6.535 |
| Polinomial Regression            |  0.08  |  447.98  | 21.166 | 16.846 |  8.334 |
| Polinomial Regression Lasso      | -0.004 |  488.715 | 22.107 | 17.445 |  8.756 |
| Polinomial Regression Ridge      |  0.079 |  448.471 | 21.177 | 16.852 |  8.339 |
| Polinomial Regression ElasticNet |  0.011 |  481.695 | 21.948 | 17.426 |  8.751 |
| RANSAC                           | -1.191 | 1066.78  | 32.662 | 25.411 |  9.846 |

---
# Clustering Algorithms Results

For the clusterization kind of algorithms I implemented K-Means and Affinity Propagation. These are not necessarily the recently most used algorithms for clustering, but they are the fundamentals from where one can study and understand clusterization problem solving and pattern recognizing.

## Clustering Results

|   Model              |Number of clusters  |     Silhouette score |
|:-------------------- | ------------------:|---------------------:| 
| K-Means              |          3         | 0.233                |
| Affinity Propagation |          7         | 0.202                |

# Conclusions
For both regression and classification types of problem, non-parametric models outperformed the parametric ones. It may be due to their hability to coope with multiple data distribution characteristics and patterns, adding flexibility to the predictions.

# Further steps
To take this essay further I will add and test more algorithms, such as XGBoost, LightGBM, Artificial Neural Networks, etc. Besides that I will implement ensemble techniques to gather the results from more than one algorithm aiming to improve their results. 
