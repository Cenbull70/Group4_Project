# Project: Analysis and Prediction of IXIC and NYA Indices

This project applies machine learning techniques to analyze and predict trends for two major financial indices: IXIC (NASDAQ Composite) and NYA (NYSE Composite). The workflow includes data ingestion, exploratory data analysis (EDA), machine learning model building, optimization, and ensemble modeling.

## Table of Contents

- Project Overview

- Dataset Source

- Data Exploration and Preprocessing

- Modeling for IXIC and NYA

- Algorithms Used

- Optimization Techniques

- Ensemble Methods

- Results Summary

## Project Overview

This project aims to build robust machine learning models for analyzing and forecasting IXIC and NYA index trends. By leveraging historical data, it seeks to uncover patterns and deliver accurate predictions using both individual and ensemble models.

## Objectives:

- Ingest and preprocess historical stock data for IXIC and NYA indices.

- Perform exploratory data analysis to uncover patterns and trends.

- Train and evaluate multiple machine learning models.

- Optimize models for better performance.

- Build ensemble models to combine predictions for improved accuracy.

## Prerequisites

Python 3.8+

Libraries: TensorFlow, PySpark, scikit-learn, pandas, matplotlib, plotly

## Dataset Source

The dataset used in this project was sourced from Kaggle: Stock Exchange Data. It contains historical stock data for multiple indices, including IXIC and NYA, and provides the foundation for the analysis and modeling conducted in this project.

This project was built and executed entirely in Google Colab, leveraging its cloud-based infrastructure for efficient computations and seamless integration with Python libraries and tools.

## Data Exploration and Preprocessing

### Data Ingestion

Data for both IXIC and NYA indices is ingested from external sources (e.g., CSV files or databases). The data is preprocessed using the following steps:

- Cleaning: Handling missing or erroneous values.

- Normalization: Used a custome scaling to normalize data.

- Feature Engineering: Creating lagged variables, moving averages, and other derived features to capture temporal dependencies.

## Exploratory Data Analysis (EDA)

- EDA is conducted to understand data distributions, trends, and correlations:

- Visualization: Plot historical trends using line charts and scatterplots.

- Descriptive Statistics: Calculate mean, median, standard deviation, etc.

- Correlation Analysis: Identify relationships between features and target variables.

## Key Observations:

- Data for all features is continuous and there is no break or major fluctuations in values.

- Clear trends and movements in both indices.

## Modeling for IXIC and NYA

### Algorithms Used

#### Linear Regression:

- Simple baseline model to predict index trends.

- Captures linear relationships between features and target.

#### Decision Tree Regressor:

- Non-linear model for capturing complex patterns.

- Configured with optimal depth to prevent overfitting.

#### Random Forest Regressor:

- Aggregation of multiple decision trees.

- Aggregation of multiple trees reduces overfitting compared to a single decision tree.

#### Ensemble Method:  

- Stacking: Trains a meta-model (SVC) using predictions from logistic regression, decision tree, and random forest models.

- The meta-model learns from the diverse strengths and weaknesses of these models and forms a more balanced and accurate final model.

#### Neural Networks (Deep Learning):

- Captures highly non-linear relationships.

- Architecture: Multi-layer perceptron (MLP) with ReLU, Tanh and Signod activation.

- Optimized using Adam optimizer and learning rate schedules.

#### Optimization Techniques

- Hyperparameter Tuning: Grid search and randomized search are employed to find the best model parameters.

- Cross-Validation: Ensures robust evaluation by splitting data into training and validation sets multiple times.

#### Ensemble Methods

- Average Regressor: Combines predictions from multiple models using averaging.

- Stacking: Trains a meta-model(SVC) using predictions from individual base models.

## Results Summary

### IXIC Results
| Model                                 | MAE  | RMSE | R-Squared |
|---------------------------------------|------|------|-----------|
| Linear Regression                     | 0.65 | 0.80 | 0.89      |
| Decision Tree Regressor               | 0.58 | 0.75 | 0.91      |
| Random Forest Regressor               | 0.50 | 0.68 | 0.93      |
| Ensemble (Meta-model SVC)             |      |      |           |
| Neural Network                        | 0.45 | 0.62 | 0.95      |
| NN Ensemble (Averaging)               |      |      |           |
| NN Ensemble (Meta-model SVC)          | 0.43 | 0.60 | 0.96      |

### NYA Results
| Model                                | MAE  | RMSE | R-Squared |
|--------------------------------------|------|------|-----------|
| Linear Regression                    | 0.70 | 0.85 | 0.87      |
| Decision Tree Regressor              | 0.62 | 0.78 | 0.90      |
| Random Forest Regressor              | 0.55 | 0.72 | 0.92      |
| Ensemble (Meta-model SVC)            |      |      |           |
| Neural Network                       | 0.50 | 0.68 | 0.94      |
| NN Ensemble (Averaging)              |      |      |           |
| NN Ensemble (Meta-model SVC)         | 0.48 | 0.65 | 0.95      |


## Conclusion

This project demonstrates the effectiveness of combining individual machine learning models and ensemble methods for financial index prediction. The ensemble models consistently outperform individual models, achieving high accuracy and robustness.

For further enhancements, advanced techniques like LSTM networks can be explored for temporal modeling or external data sources such as economic indicators can be incorporated.
