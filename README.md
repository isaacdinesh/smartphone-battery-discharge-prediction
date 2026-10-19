# Smartphone Battery Discharge Prediction

Machine learning analysis of Android smartphone battery discharge rates using the GreenHub dataset.

This project compares Linear Regression, Random Forest and XGBoost to estimate battery discharge rates from historical device telemetry.

## Dataset
GreenHub Farmer: https://www.kaggle.com/datasets/hmatalonga/greenhub-farmer

## Models
- Linear Regression
- Random Forest Regression
- XGBoost Regression

## Best Result
Tuned Random Forest: MAE = 5.9054, RMSE = 8.4529, R² = 0.3116.

The model estimates discharge rates from completed sessions and is not a real-time battery-life forecasting system.
