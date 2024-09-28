import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import keras

# Load the dataset
dataset = pd.read_csv("data/output.csv", header=0).dropna()
dataset = dataset[['price', 'ma7', '26ema', '12ema', 'MACD', 'ema', 'momentum']]
dataset = dataset[::-1].reset_index(drop=True)

# Check for NaNs in the original dataset
if dataset.isnull().values.any():
    print("NaN values found in the original dataset.")

train_data = dataset.iloc[:, 1:].values
target_data = dataset.iloc[:, 0].values

# Set print options for NumPy
np.set_printoptions(suppress=True, precision=6)

# Standardize the input features
scaler_X = StandardScaler()
data_scaled = scaler_X.fit_transform(train_data)

# Standardize the target variable
scaler_y = StandardScaler()
target_data = target_data.reshape(-1, 1)  # Reshape for scaler
target_data_scaled = scaler_y.fit_transform(target_data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target_data_scaled, test_size=0.3, random_state=42)

# Check for NaNs after splitting
if np.isnan(X_train).any() or np.isnan(y_train).any():
    print("NaN values found in training data after split.")

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 6

input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation="gelu")(input_layer)
decoder = keras.layers.Dense(input_dim, activation="relu")(encoder)
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Use encoder part of the autoencoder for feature selection
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

# Check for NaNs in encoded features
if np.isnan(encoded_features_train).any() or np.isnan(encoded_features_test).any():
    print("NaN values found in encoded features.")

# Autocorrelation
def autocorrelation(data, lag=1):
    return [data.iloc[:, i].autocorr(lag) for i in range(data.shape[1])]

autocorr_values = autocorrelation(pd.DataFrame(encoded_features_train), lag=1)

# Convert autocorr_values to a NumPy array
weights = np.array(autocorr_values)

# Apply the weights to your input features
weighted_encoded_features_train = encoded_features_train * weights

# Check for NaNs in weighted features
if np.isnan(weighted_encoded_features_train).any() or np.isnan(y_train).any():
    print("NaN values found in weighted features.")

# Perform regression using Linear Regression
# regressor = LinearRegression()
# regressor.fit(weighted_encoded_features_train, y_train)

# Predict on the test set
# weighted_encoded_features_test = encoded_features_test * weights
# y_pred = regressor.predict(weighted_encoded_features_test)

# training_days = int(train_data.shape[0]*.7)
# train_data_test = train_data[training_days:]
# regressor = LinearRegression()
# regressor.fit(train_data[:training_days], y_train)
# y_pred_unscaled = regressor.predict(train_data_test)
# target_data_test = target_data[training_days:]

# Inverse transform the predicted values and actual values
# y_pred_original = scaler_y.inverse_transform(y_pred)
# y_test_original = scaler_y.inverse_transform(y_test)

# Calculate Mean Squared Error (MSE) on the original scale
# mse_original = mean_squared_error(target_data_test, y_pred_unscaled)
# print(f'Mean Squared Error (Original Scale): {mse_original}')

# Plotting the results
# plt.figure(figsize=(10, 5))
# plt.plot(target_data_test, label='True Values', color='blue')
# plt.plot(y_pred_unscaled, label='Predicted Values', color='orange', linestyle='--')
# plt.title('True vs Predicted Values (Original Scale)')
# plt.xlabel('Sample Index')
# plt.ylabel('Target Value')
# plt.legend()
# plt.show()
