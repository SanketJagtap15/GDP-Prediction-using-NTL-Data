
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load the data
# Assuming files are loaded locally as gddp_data.csv and ntl_data.csv
gddp_data = pd.read_csv('gddp_data.csv')
ntl_data = pd.read_csv('ntl_data.csv')

# Step 2: Data Preprocessing
# Rename columns to prepare for merging
gddp_data.rename(columns={'year': 'District'}, inplace=True)
ntl_data.rename(columns={'Name of District': 'District'}, inplace=True)

# Merge GDDP and NTL datasets on 'District' for overlapping years 1999-2013
years_common = [str(year) for year in range(1999, 2014)]
merged_data = pd.merge(gddp_data[['District'] + years_common], 
                       ntl_data[['District'] + years_common], 
                       on='District', suffixes=('_GDDP', '_NTL'))

# Remove duplicate rows by keeping only the first instance
merged_data = merged_data.drop_duplicates(subset='District', keep='first')

# Extract and flatten data for machine learning
X = merged_data[[f"{year}_NTL" for year in range(1999, 2014)]]
y = merged_data[[f"{year}_GDDP" for year in range(1999, 2014)]]

# Flatten data into a long format for training
X_flattened = X.values.flatten().reshape(-1, 1)
y_flattened = y.values.flatten()

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_flattened, test_size=0.2, random_state=42)

# Step 4: Ensemble Model Training
# Hyperparameter-tuned models for ensemble
tuned_rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
tuned_gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
tuned_ensemble_model = VotingRegressor([('lr', LinearRegression()), ('rf', tuned_rf), ('gb', tuned_gb)])
tuned_ensemble_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = tuned_ensemble_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ensemble Model MSE: {mse}")
print(f"Ensemble Model R2: {r2}")

# Step 6: LSTM Model Training
# Reshape data for LSTM model (samples, timesteps, features)
X_lstm = X.values.reshape((X.shape[0], X.shape[1], 1))
y_lstm = y.values.flatten()

# Scale data to [0, 1] for LSTM input
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_lstm_scaled = scaler_X.fit_transform(X.values).reshape((X.shape[0], X.shape[1], 1))
y_lstm_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train-Test Split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm_scaled, y_lstm_scaled, test_size=0.2, random_state=42)

# Define and train LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=1)

# Predict and evaluate LSTM model
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_test_lstm_rescaled = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1))

lstm_mse = mean_squared_error(y_test_lstm_rescaled, y_pred_lstm)
lstm_r2 = r2_score(y_test_lstm_rescaled, y_pred_lstm)
print(f"LSTM Model MSE: {lstm_mse}")
print(f"LSTM Model R2: {lstm_r2}")

# Step 7: Visualization
# Plotting actual vs predicted values for a random sample
sample_indices = random.sample(range(len(y_test)), 10)
y_test_sample = y_test[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.figure(figsize=(12, 6))
plt.plot(y_test_sample, label='Actual GDDP', marker='o', linestyle='-')
plt.plot(y_pred_sample, label='Predicted GDDP', marker='x', linestyle='--')
plt.xlabel('Sample Districts')
plt.ylabel('GDDP')
plt.title('Actual vs Predicted GDDP for Random Sample of Districts')
plt.legend()
plt.show()
