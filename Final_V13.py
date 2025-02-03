# JET SAW JUN JIE 1231303401
# LIAW RUO SHAN 1231302935

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Timer start
start_time = time.time()

# Read and preprocess the dataset
yield_df = pd.read_csv("https://raw.githubusercontent.com/Jetsaw/Ai_Assigment/refs/heads/main/yield_df.csv")
print("=== Yield Dataset ===")
print(yield_df.head(5))

print("Checking for missing values:")
print(yield_df.isna().sum())

print("Dataset Information:")
print(yield_df.info())

# Data cleaning
yield_df = yield_df.drop(columns=['Unnamed: 0'])

# Convert numeric columns dtype
numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
yield_df[numeric_cols] = yield_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Encode 'Area' column using Label Encoding
le = LabelEncoder()
yield_df['Area'] = le.fit_transform(yield_df['Area'])

# Removing outliers based on 'hg/ha_yield'
z_scores = stats.zscore(yield_df['hg/ha_yield'])
yield_df = yield_df[np.abs(z_scores) < 3]

# One-hot encoding for categorical variables
yield_df = pd.get_dummies(yield_df)
print(yield_df.head(5))

# Split the data into features and target
X = yield_df.drop(columns=['hg/ha_yield'])
y = yield_df['hg/ha_yield']

# Feature selection
selector = SelectKBest(k=10)
X = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define machine learning models
models = {
    'lnr': LinearRegression(),
    'knn': KNeighborsRegressor(),
    'dtr': DecisionTreeRegressor(random_state=42),
    'rfr': RandomForestRegressor(random_state=42),
    'gbr': GradientBoostingRegressor(random_state=42),
    'mlp': MLPRegressor(random_state=42),
    'svr': SVR()
}

# Define Feature Scaling
scalers = {
    'NO': None,
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler()
}
results = []
for m in scalers:
    if scalers[m]:
        X_train_scaled = scalers[m].fit_transform(X_train)
        X_test_scaled = scalers[m].transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    print(f"\nWith {m} Scaling:")

    for n in models:
        models[n].fit(X_train_scaled, y_train)
        score = models[n].score(X_test_scaled, y_test)
        print(f"{n}: {score:.3%}")

        # Store results
        results.append({
            'Scaler': m,
            'Model': n,
            'R2 Score': score
        })

# Store the results in DataFrame
results_df = pd.DataFrame(results)

# Print the results
print("=== Model Performance ===")
print(results_df)

# Robust Scaler is selected for hyperparameter tuning
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the results
print("\nWithout Hyperparameter Tuning:\n")
plt.figure(figsize=(14, 10))


for i, n in enumerate(models, start=1):
    models[n].fit(X_train_scaled, y_train)
    y_pred = models[n].predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    score = models[n].score(X_test_scaled, y_test)

    plt.subplot(3, 3, i)
    plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{n} Predicted vs Actual\nR2: {score:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}')
    print(f"R2 score ({n}): {score:.3%}")
    print(f"Root Mean Squared Error ({n}): {rmse:.3f}")
    print(f"Mean Absolute Error ({n}): {mae:.3f}\n")

plt.tight_layout()
plt.show()

# Hyperparameter Tuning
kf = KFold(n_splits=3, shuffle=True, random_state=42)

param_grids = {
    'lnr': {'copy_X': [True,False], 'fit_intercept': [True,False], 'n_jobs': [1,5,10,15,None], 'positive': [True,False]},
    'knn': {'leaf_size': [5, 10, 15], 'n_jobs': [1, 5, 10, 15, None], 'n_neighbors': [20, 50, 80, 100], 'weights': ['uniform', 'distance']},
    'dtr': {'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4]},
    'rfr': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    'gbr': {'alpha': [0.1, 0.5, 0.9], 'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'mlp': {'hidden_layer_sizes': [10, 30, 50, 70], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.00005, 0.0001, 0.0005]},
    'svr': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}
}

best_params = {}

for n in param_grids:
    grid_search = GridSearchCV(estimator=models[n], param_grid=param_grids[n], cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_params[n] = grid_search.best_params_

print("Best Parameters:")
for n in best_params:
    print(f"{n}: {best_params[n]}")

models_tuned = {
    'lnr': LinearRegression(**best_params['lnr']),
    'knn': KNeighborsRegressor(**best_params['knn']),
    'dtr': DecisionTreeRegressor(random_state=42, **best_params['dtr']),
    'rfr': RandomForestRegressor(random_state=42, **best_params['rfr']),
    'gbr': GradientBoostingRegressor(random_state=42, **best_params['gbr']),
    'mlp': MLPRegressor(random_state=42, **best_params['mlp']),
    'svr': SVR(**best_params['svr'])
}

print("\nWith Hyperparameter Tuning:\n")

plt.figure(figsize=(14, 10))

for i, n in enumerate(models_tuned, start=1):
    models_tuned[n].fit(X_train_scaled, y_train)
    y_pred = models_tuned[n].predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    score = models_tuned[n].score(X_test_scaled, y_test)

    plt.subplot(3, 3, i)
    plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{n} Predicted vs Actual\nR2: {score:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}')
    print(f"R2 score ({n}): {score:.3%}")
    print(f"Root Mean Squared Error ({n}): {rmse:.3f}")
    print(f"Mean Absolute Error ({n}): {mae:.3f}\n")

# Display the plot
plt.tight_layout()
plt.show()

# Timer end
end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
