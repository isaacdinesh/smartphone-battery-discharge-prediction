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
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# --------------------------------------------------
# LOAD FINAL MODELLING DATASET
# --------------------------------------------------

data_folder = Path('data')

df = pd.read_parquet(
    data_folder / 'model_sessions.parquet'
)


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


# --------------------------------------------------
# REPRODUCE EXACT TRAIN / TEST SPLIT
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


print('Train Sessions:', len(X_train))
print('Test Sessions:', len(X_test))


# --------------------------------------------------
# FINAL RANDOM FOREST
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


# --------------------------------------------------
# FINAL XGBOOST
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


# --------------------------------------------------
# TRAIN FINAL MODELS
# --------------------------------------------------

final_rf_model.fit(
    X_train,
    y_train
)

final_xgb_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# TEST-SET PREDICTIONS
# --------------------------------------------------

rf_predictions = final_rf_model.predict(
    X_test
)

xgb_predictions = final_xgb_model.predict(
    X_test
)


# --------------------------------------------------
# CALCULATE FINAL METRICS
# --------------------------------------------------

def calculate_metrics(
    actual,
    predicted
):
    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


rf_mae, rf_rmse, rf_r2 = calculate_metrics(
    y_test,
    rf_predictions
)

xgb_mae, xgb_rmse, xgb_r2 = calculate_metrics(
    y_test,
    xgb_predictions
)


print('\n--------------------------------')
print('FINAL MODEL RESULTS')
print('--------------------------------')

print('\nRandom Forest')
print(f'MAE:  {rf_mae:.4f}')
print(f'RMSE: {rf_rmse:.4f}')
print(f'R²:   {rf_r2:.4f}')

print('\nXGBoost')
print(f'MAE:  {xgb_mae:.4f}')
print(f'RMSE: {xgb_rmse:.4f}')
print(f'R²:   {xgb_r2:.4f}')


# --------------------------------------------------
# RESIDUALS
# --------------------------------------------------

rf_residuals = (
    y_test.to_numpy()
    - rf_predictions
)

xgb_residuals = (
    y_test.to_numpy()
    - xgb_predictions
)


print('\n--------------------------------')
print('RESIDUAL STATISTICS')
print('--------------------------------')

print('\nRandom Forest Residuals:')
print(
    pd.Series(rf_residuals).describe()
)

print('\nXGBoost Residuals:')
print(
    pd.Series(xgb_residuals).describe()
)


print('\nMean RF Residual:')
print(
    f'{np.mean(rf_residuals):.4f}'
)

print('\nMean XGBoost Residual:')
print(
    f'{np.mean(xgb_residuals):.4f}'
)


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

output_folder = Path('results')
output_folder.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# ACTUAL VS PREDICTED - RANDOM FOREST
# --------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.25,
    s=15
)

plot_min = min(
    y_test.min(),
    rf_predictions.min()
)

plot_max = max(
    y_test.max(),
    rf_predictions.max()
)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle='--'
)

plt.xlabel(
    'Actual Discharge Rate (% points/hour)'
)

plt.ylabel(
    'Predicted Discharge Rate (% points/hour)'
)

plt.title(
    'Random Forest: Actual vs Predicted Discharge Rate'
)

plt.tight_layout()

plt.savefig(
    output_folder / 'rf_actual_vs_predicted.png',
    dpi=300
)

plt.close()


# --------------------------------------------------
# ACTUAL VS PREDICTED - XGBOOST
# --------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    y_test,
    xgb_predictions,
    alpha=0.25,
    s=15
)

plot_min = min(
    y_test.min(),
    xgb_predictions.min()
)

plot_max = max(
    y_test.max(),
    xgb_predictions.max()
)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle='--'
)

plt.xlabel(
    'Actual Discharge Rate (% points/hour)'
)

plt.ylabel(
    'Predicted Discharge Rate (% points/hour)'
)

plt.title(
    'XGBoost: Actual vs Predicted Discharge Rate'
)

plt.tight_layout()

plt.savefig(
    output_folder / 'xgb_actual_vs_predicted.png',
    dpi=300
)

plt.close()


# --------------------------------------------------
# RANDOM FOREST RESIDUAL PLOT
# --------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    rf_predictions,
    rf_residuals,
    alpha=0.25,
    s=15
)

plt.axhline(
    y=0,
    linestyle='--'
)

plt.xlabel(
    'Predicted Discharge Rate (% points/hour)'
)

plt.ylabel(
    'Residual (Actual - Predicted)'
)

plt.title(
    'Random Forest: Residuals vs Predicted Values'
)

plt.tight_layout()

plt.savefig(
    output_folder / 'rf_residual_plot.png',
    dpi=300
)

plt.close()


# --------------------------------------------------
# XGBOOST RESIDUAL PLOT
# --------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    xgb_predictions,
    xgb_residuals,
    alpha=0.25,
    s=15
)

plt.axhline(
    y=0,
    linestyle='--'
)

plt.xlabel(
    'Predicted Discharge Rate (% points/hour)'
)

plt.ylabel(
    'Residual (Actual - Predicted)'
)

plt.title(
    'XGBoost: Residuals vs Predicted Values'
)

plt.tight_layout()

plt.savefig(
    output_folder / 'xgb_residual_plot.png',
    dpi=300
)

plt.close()


print(
    '\nDiagnostic plots saved to results folder.'
)


# --------------------------------------------------
# ERROR BY ACTUAL DISCHARGE-RATE RANGE
# --------------------------------------------------

error_analysis = pd.DataFrame({
    'actual': y_test.to_numpy(),
    'rf_prediction': rf_predictions,
    'xgb_prediction': xgb_predictions
})

error_analysis['discharge_group'] = pd.cut(
    error_analysis['actual'],
    bins=[
        0,
        10,
        20,
        30,
        40,
        np.inf
    ],
    labels=[
        '0-10',
        '10-20',
        '20-30',
        '30-40',
        '40+'
    ],
    right=False
)


def group_error(group):
    rf_mae = mean_absolute_error(
        group['actual'],
        group['rf_prediction']
    )

    xgb_mae = mean_absolute_error(
        group['actual'],
        group['xgb_prediction']
    )

    return pd.Series({
        'sessions': len(group),
        'rf_mae': rf_mae,
        'xgb_mae': xgb_mae,
        'mean_actual': group['actual'].mean(),
        'mean_rf_prediction':
            group['rf_prediction'].mean(),
        'mean_xgb_prediction':
            group['xgb_prediction'].mean()
    })


group_results = (
    error_analysis
    .groupby(
        'discharge_group',
        observed=True
    )
    .apply(
        group_error,
        include_groups=False
    )
)


print('\n--------------------------------')
print('ERROR BY ACTUAL DISCHARGE RATE')
print('--------------------------------')

print(group_results)


# --------------------------------------------------
# PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

print('\n--------------------------------')
print('PERMUTATION FEATURE IMPORTANCE')
print('--------------------------------')


# Random Forest
rf_importance = permutation_importance(
    final_rf_model,
    X_test,
    y_test,
    scoring='neg_mean_absolute_error',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


rf_importance_df = pd.DataFrame({
    'feature': features,
    'importance':
        rf_importance.importances_mean,
    'std':
        rf_importance.importances_std
}).sort_values(
    'importance',
    ascending=False
)


# XGBoost
xgb_importance = permutation_importance(
    final_xgb_model,
    X_test,
    y_test,
    scoring='neg_mean_absolute_error',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


xgb_importance_df = pd.DataFrame({
    'feature': features,
    'importance':
        xgb_importance.importances_mean,
    'std':
        xgb_importance.importances_std
}).sort_values(
    'importance',
    ascending=False
)


print('\nRandom Forest:')
print(
    rf_importance_df.to_string(
        index=False
    )
)

print('\nXGBoost:')
print(
    xgb_importance_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE PLOT
# --------------------------------------------------

rf_plot_data = (
    rf_importance_df
    .sort_values(
        'importance',
        ascending=True
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.barh(
    rf_plot_data['feature'],
    rf_plot_data['importance'],
    xerr=rf_plot_data['std']
)

plt.axvline(
    x=0,
    linestyle='--'
)

plt.xlabel(
    'Increase in MAE After Permutation'
)

plt.ylabel(
    'Predictor'
)

plt.title(
    'Random Forest Permutation Feature Importance'
)

plt.tight_layout()

plt.savefig(
    output_folder /
    'rf_permutation_importance.png',
    dpi=300
)

plt.close()


print(
    '\nFeature importance plot saved.'
)