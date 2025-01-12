import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats

# Timer start
start_time = time.time()

# Read and preprocess the dataset
yield_df = read_csv("https://raw.githubusercontent.com/Jetsaw/Ai_Assigment/refs/heads/main/yield_df.csv")
print("\n=== Yield Dataset Summary ===")
print(yield_df.head())

# Data cleaning
numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
yield_df[numeric_cols] = yield_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
yield_df.dropna(inplace=True)

# Removing outliers based on 'hg/ha_yield'
z_scores = stats.zscore(yield_df['hg/ha_yield'])
yield_df = yield_df[np.abs(z_scores) < 3]

# Encode categorical column
le = LabelEncoder()
yield_df['Area'] = le.fit_transform(yield_df['Area'])

# Plot: Mean Yield Over Time
temp_data = yield_df.groupby(['Year', 'Item'])[['hg/ha_yield']].mean()
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle('Mean Harvested Value (1961-2019)', fontsize=16)
temp_data['hg/ha_yield'].unstack().plot(ax=ax)
ax.set_ylabel('Mean Yield (hg/ha)', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
plt.show()

# Splitting data
X = yield_df.drop(columns=['hg/ha_yield'])
y = yield_df['hg/ha_yield']
X = pd.get_dummies(X, columns=['Item'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define scalers
scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer()
}

# Define models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

# Evaluate models
results = []
for scaler_name, scaler in scalers.items():
    # Apply scaling
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Evaluate each model
    for model_name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        r2 = r2_score(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Store results
        results.append({
            'Scaler': scaler_name,
            'Model': model_name,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Identify best results
best_r2 = results_df.loc[results_df['R2'].idxmax()]
best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
best_mae = results_df.loc[results_df['MAE'].idxmin()]

# Display results
print("\n=== Model Performance ===")
print(results_df.sort_values(by='R2', ascending=False).to_string(index=False))

# Highlight best results
print("\n=== Best Combinations ===")
print(f"Best R2 Score: {best_r2['Model']} with {best_r2['Scaler']} (R2 = {best_r2['R2']:.3f})")
print(f"Best RMSE: {best_rmse['Model']} with {best_rmse['Scaler']} (RMSE = {best_rmse['RMSE']:.3f})")
print(f"Best MAE: {best_mae['Model']} with {best_mae['Scaler']} (MAE = {best_mae['MAE']:.3f})")

# Scatter Plot: Actual vs Predicted (Best R2 Model)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, models[best_r2['Model']].predict(X_test_pca), color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title(f'Actual vs Predicted ({best_r2["Model"]})', fontsize=14)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.show()

# Timer end
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.2f} seconds")
