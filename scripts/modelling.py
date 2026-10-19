import sys
from pathlib import Path

class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            output.flush()

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

log_file = open(
    results_folder / f"{Path(__file__).stem}_output.txt",
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(sys.stdout, log_file)


import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV
)

from scipy.stats import randint

import numpy as np
import time

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# --------------------------------------------------
# LOAD FINAL MODELLING DATASET
# --------------------------------------------------

data_folder = Path('data')

df = pd.read_parquet(
    data_folder / 'model_sessions.parquet'
)

print('Dataset Shape:')
print(df.shape)

print('\nUnique Devices:')
print(df['device_id'].nunique())


# --------------------------------------------------
# DEFINE PREDICTORS AND TARGET
# --------------------------------------------------

features = [
    'avg_cpu_usage',
    'avg_temperature',
    'screen_on_proportion',
    'avg_manual_brightness',
    'auto_brightness_proportion',
    'wifi_proportion',
    'mobile_proportion',
    'power_saver_proportion'
]

target = 'discharge_rate'

X = df[features].copy()
y = df[target].copy()

groups = df['device_id']


print('\nPredictors:')
print(features)

print('\nTarget:')
print(target)

print('\nX Shape:')
print(X.shape)

print('\ny Shape:')
print(y.shape)


# --------------------------------------------------
# DEVICE-LEVEL TRAIN / TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]


# --------------------------------------------------
# CHECK SPLIT
# --------------------------------------------------

print('\nTrain Sessions:')
print(len(X_train))

print('\nTest Sessions:')
print(len(X_test))

print('\nTrain Devices:')
print(train_groups.nunique())

print('\nTest Devices:')
print(test_groups.nunique())

overlap = set(train_groups).intersection(
    set(test_groups)
)

print('\nDevices Appearing in Both Train and Test:')
print(len(overlap))


print('\nTraining Target Statistics:')
print(y_train.describe())

print('\nTesting Target Statistics:')
print(y_test.describe())


print('\nTraining Missing Values:')
print(X_train.isna().sum())

print('\nTesting Missing Values:')
print(X_test.isna().sum())


# --------------------------------------------------
# LINEAR REGRESSION
# --------------------------------------------------

linear_model = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'scaler',
            StandardScaler()
        ),
        (
            'model',
            LinearRegression()
        )
    ]
)


# Train model and record training time
start_time = time.perf_counter()

linear_model.fit(
    X_train,
    y_train
)

linear_training_time = (
    time.perf_counter() - start_time
)


# Make predictions and record prediction time
start_time = time.perf_counter()

linear_predictions = linear_model.predict(
    X_test
)

linear_prediction_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# EVALUATE LINEAR REGRESSION
# --------------------------------------------------

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print('\n--------------------------------')
print('LINEAR REGRESSION RESULTS')
print('--------------------------------')

print(f'MAE:  {linear_mae:.4f}')
print(f'RMSE: {linear_rmse:.4f}')
print(f'R²:   {linear_r2:.4f}')

print(
    f'Training Time: '
    f'{linear_training_time:.4f} seconds'
)

print(
    f'Prediction Time: '
    f'{linear_prediction_time:.4f} seconds'
)


# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Train model and record training time
start_time = time.perf_counter()

random_forest_model.fit(
    X_train,
    y_train
)

rf_training_time = (
    time.perf_counter() - start_time
)


# Make predictions and record prediction time
start_time = time.perf_counter()

rf_predictions = random_forest_model.predict(
    X_test
)

rf_prediction_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# EVALUATE RANDOM FOREST
# --------------------------------------------------

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print('\n--------------------------------')
print('RANDOM FOREST RESULTS')
print('--------------------------------')

print(f'MAE:  {rf_mae:.4f}')
print(f'RMSE: {rf_rmse:.4f}')
print(f'R²:   {rf_r2:.4f}')

print(
    f'Training Time: '
    f'{rf_training_time:.4f} seconds'
)

print(
    f'Prediction Time: '
    f'{rf_prediction_time:.4f} seconds'
)


# --------------------------------------------------
# XGBOOST
# --------------------------------------------------

xgboost_model = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Train model and record training time
start_time = time.perf_counter()

xgboost_model.fit(
    X_train,
    y_train
)

xgb_training_time = (
    time.perf_counter() - start_time
)


# Make predictions and record prediction time
start_time = time.perf_counter()

xgb_predictions = xgboost_model.predict(
    X_test
)

xgb_prediction_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# EVALUATE XGBOOST
# --------------------------------------------------

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)


print('\n--------------------------------')
print('XGBOOST RESULTS')
print('--------------------------------')

print(f'MAE:  {xgb_mae:.4f}')
print(f'RMSE: {xgb_rmse:.4f}')
print(f'R²:   {xgb_r2:.4f}')

print(
    f'Training Time: '
    f'{xgb_training_time:.4f} seconds'
)

print(
    f'Prediction Time: '
    f'{xgb_prediction_time:.4f} seconds'
)



# --------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING
# --------------------------------------------------

print('\n--------------------------------')
print('RANDOM FOREST TUNING')
print('--------------------------------')

# Group-aware cross-validation
group_cv = GroupKFold(
    n_splits=5
)


# Random Forest pipeline
rf_tuning_pipeline = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )
        )
    ]
)


# Hyperparameter search space
rf_parameter_space = {
    'model__n_estimators': randint(150, 501),
    'model__max_depth': [
        None,
        10,
        15,
        20,
        30
    ],
    'model__min_samples_leaf': randint(1, 11),
    'model__max_features': [
        0.5,
        0.75,
        1.0,
        'sqrt'
    ]
}


# Randomised search
rf_search = RandomizedSearchCV(
    estimator=rf_tuning_pipeline,
    param_distributions=rf_parameter_space,
    n_iter=10,
    scoring='neg_root_mean_squared_error',
    cv=group_cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)


# Train using ONLY the training set
start_time = time.perf_counter()

rf_search.fit(
    X_train,
    y_train,
    groups=train_groups
)

rf_tuning_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# RANDOM FOREST TUNING RESULTS
# --------------------------------------------------

print('\nBest Random Forest Parameters:')
print(
    rf_search.best_params_
)

print('\nBest Cross-Validation RMSE:')
print(
    f'{-rf_search.best_score_:.4f}'
)

print(
    '\nHyperparameter Tuning Time: '
    f'{rf_tuning_time:.4f} seconds'
)


# Group-aware cross-validation
group_cv = GroupKFold(
    n_splits=5
)


# --------------------------------------------------
# XGBOOST HYPERPARAMETER TUNING
# --------------------------------------------------

print('\n--------------------------------')
print('XGBOOST TUNING')
print('--------------------------------')


# XGBoost pipeline
xgb_tuning_pipeline = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=1
            )
        )
    ]
)


# Hyperparameter search space
xgb_parameter_space = {
    'model__n_estimators': [
        200,
        300,
        400,
        500
    ],
    'model__learning_rate': [
        0.03,
        0.05,
        0.1
    ],
    'model__max_depth': [
        3,
        5,
        7
    ],
    'model__subsample': [
        0.7,
        0.85,
        1.0
    ],
    'model__colsample_bytree': [
        0.7,
        0.85,
        1.0
    ]
}


# Randomised search
xgb_search = RandomizedSearchCV(
    estimator=xgb_tuning_pipeline,
    param_distributions=xgb_parameter_space,
    n_iter=10,
    scoring='neg_root_mean_squared_error',
    cv=group_cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)


# Train using ONLY the training set
start_time = time.perf_counter()

xgb_search.fit(
    X_train,
    y_train,
    groups=train_groups
)

xgb_tuning_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# XGBOOST TUNING RESULTS
# --------------------------------------------------

print('\nBest XGBoost Parameters:')
print(
    xgb_search.best_params_
)

print('\nBest Cross-Validation RMSE:')
print(
    f'{-xgb_search.best_score_:.4f}'
)

print(
    '\nHyperparameter Tuning Time: '
    f'{xgb_tuning_time:.4f} seconds'
)



# --------------------------------------------------
# FINAL TUNED MODELS
# --------------------------------------------------

print('\n================================')
print('FINAL TUNED MODEL EVALUATION')
print('================================')


# --------------------------------------------------
# TUNED RANDOM FOREST
# --------------------------------------------------

final_rf_model = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            RandomForestRegressor(
                n_estimators=393,
                max_depth=15,
                max_features='sqrt',
                min_samples_leaf=7,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


start_time = time.perf_counter()

final_rf_model.fit(
    X_train,
    y_train
)

final_rf_training_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

final_rf_predictions = final_rf_model.predict(
    X_test
)

final_rf_prediction_time = (
    time.perf_counter() - start_time
)


final_rf_mae = mean_absolute_error(
    y_test,
    final_rf_predictions
)

final_rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_rf_predictions
    )
)

final_rf_r2 = r2_score(
    y_test,
    final_rf_predictions
)


# --------------------------------------------------
# TUNED XGBOOST
# --------------------------------------------------

final_xgb_model = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median',
                add_indicator=True
            )
        ),
        (
            'model',
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


start_time = time.perf_counter()

final_xgb_model.fit(
    X_train,
    y_train
)

final_xgb_training_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

final_xgb_predictions = final_xgb_model.predict(
    X_test
)

final_xgb_prediction_time = (
    time.perf_counter() - start_time
)


final_xgb_mae = mean_absolute_error(
    y_test,
    final_xgb_predictions
)

final_xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_xgb_predictions
    )
)

final_xgb_r2 = r2_score(
    y_test,
    final_xgb_predictions
)


# --------------------------------------------------
# PRINT FINAL RESULTS
# --------------------------------------------------

print('\nTUNED RANDOM FOREST')
print('--------------------------------')
print(f'MAE:  {final_rf_mae:.4f}')
print(f'RMSE: {final_rf_rmse:.4f}')
print(f'R²:   {final_rf_r2:.4f}')
print(
    f'Training Time: '
    f'{final_rf_training_time:.4f} seconds'
)
print(
    f'Prediction Time: '
    f'{final_rf_prediction_time:.4f} seconds'
)


print('\nTUNED XGBOOST')
print('--------------------------------')
print(f'MAE:  {final_xgb_mae:.4f}')
print(f'RMSE: {final_xgb_rmse:.4f}')
print(f'R²:   {final_xgb_r2:.4f}')
print(
    f'Training Time: '
    f'{final_xgb_training_time:.4f} seconds'
)
print(
    f'Prediction Time: '
    f'{final_xgb_prediction_time:.4f} seconds'
)