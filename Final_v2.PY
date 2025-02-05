import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pandas import read_csv
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy import stats

# Timer start
start_time = time.time()

# Read and display the yield dataset
yield_df = read_csv("https://raw.githubusercontent.com/Jetsaw/Ai_Assigment/refs/heads/main/yield_df.csv")
print("\n=== Yield Dataset ===")
print(yield_df.head())
print("\nChecking for missing values:")
print(yield_df.isna().sum())
print("\nDataset Information:")
yield_df.info()

# Convert numeric columns to appropriate dtype
numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
yield_df[numeric_cols] = yield_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Encode 'Area' column using Label Encoding
le = LabelEncoder()
yield_df['Area'] = le.fit_transform(yield_df['Area'])

# Plot the data
temp_data = yield_df.groupby(['Year', 'Item'])[['hg/ha_yield']].mean()
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle('Mean Harvested Value (1961-2019)')
temp_data['hg/ha_yield'].unstack().plot(ax=ax)
ax.set_ylabel('Mean Yield (hg/ha)')
ax.set_xlabel('Year')
plt.show()

# Data cleaning
yield_df = yield_df.dropna()

# Removing outliers based on 'hg/ha_yield'
z_scores = stats.zscore(yield_df['hg/ha_yield'])
yield_df = yield_df[np.abs(z_scores) < 3]

# Split the data into features and target
X = yield_df.drop(columns=['hg/ha_yield'])
y = yield_df['hg/ha_yield']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Item'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

############Feature & Data Scaling####################

# Data scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

-------------------------------------------------------

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=500, learning_rate=0.1,max_depth=6)
xgb.fit(X_train_pca, y_train)
y_pred_xgb = xgb.predict(X_test_pca)
xgb_r2 = r2_score(y_test, y_pred_xgb)
print("XGBoost R2 Score:", xgb_r2)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)
print("Best Parameters for Random Forest:", grid_search.best_params_)

# Random Forest with optimized parameters
rf = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest R2 Score:", rf_r2)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest MAE:", rf_mae)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_pca, y_train)
y_pred_lr = lin_reg.predict(X_test_pca)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print("Linear Regression R2 Score:", lr_r2)
print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression MAE:", lr_mae)

#K NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)
knn_r2 = r2_score(y_test, y_pred_knn)
knn_rmse = sqrt(mean_squared_error(y_test, y_pred_knn))
knn_mae = mean_absolute_error(y_test, y_pred_knn)

print("KNN R2 Score:", knn_r2)
print("KNN RMSE:", knn_rmse)
print("KNN MAE:", knn_mae)

#Neural Network
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
mlp.fit(X_train_pca, y_train)
y_pred_mlp = mlp.predict(X_test_pca)
mlp_r2 = r2_score(y_test, y_pred_mlp)
mlp_rmse = sqrt(mean_squared_error(y_test, y_pred_mlp))

print("MLP R2 Score:", mlp_r2)
print("MLP RMSE:", mlp_rmse)





# Model Performance Summary
results = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest', 'Linear Regression'],
    'R2 Score': [xgb_r2, rf_r2, lr_r2],
    'RMSE': [None, rf_rmse, lr_rmse],
    'MAE': [None, rf_mae, lr_mae]
})
print("\n=== Model Performance Summary ===")
print(results)

# Scatter Plot: Actual vs Predicted (Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Scatter Plot: Actual vs Predicted (Linear Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# Timer end 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
