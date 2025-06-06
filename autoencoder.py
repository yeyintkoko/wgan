import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models
from keras.optimizers import Adam

def load_data():
    # Load the dataset
    dataset = pd.read_csv("data/output.csv", header=0).dropna()
    selected_columns = ['price', 'ma7', '26ema', '12ema', 'upper_band', 'lower_band', 'ema', 'momentum']
    dataset = dataset[selected_columns]
    dataset = dataset[::-1].reset_index(drop=True)

    # Reduce the data to 50% for faster turning
    # num_reduce_days = int(dataset.shape[0]*.5)
    # dataset = dataset.iloc[num_reduce_days:]

    # Check for NaNs in the original dataset
    if dataset.isnull().values.any():
        print("NaN values found in the original dataset.")

    # Set print options for NumPy
    np.set_printoptions(suppress=True, precision=6)

    num_training_days = int(dataset.shape[0]*.9)

    data_train = dataset.iloc[:num_training_days].values
    data_test = dataset.iloc[num_training_days:].values

    # Split the data into training and test sets
    X_train = data_train[:, :]  # all features except the first one (price)
    y_train = data_train[:, 0]    # only the first one (price)
    X_test = data_test[:, :]      # all features except the first one
    y_test = data_test[:, 0]       # only the first one (price)
    
    # Standardize the input features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)  # Use transform, not fit_transform

    # Standardize the target variable
    scaler_y = StandardScaler()
    y_train_reshaped = y_train.reshape(-1, 1)  # Reshape for scaler
    y_test_reshaped = y_test.reshape(-1, 1)  # Reshape for scaler
    y_train = scaler_y.fit_transform(y_train_reshaped).flatten()
    y_test = scaler_y.transform(y_test_reshaped).flatten()  # Use transform, not fit_transform

    return (X_train, y_train), (X_test, y_test), (scaler_X, scaler_y), num_training_days, selected_columns

def build_stacked_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    encoded = layers.Dense(64, activation='gelu')(input_layer)
    encoded = layers.Dense(input_shape[0], activation='linear')(encoded)
    
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(1, activation='linear')(decoded)
    
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return autoencoder, encoder

(X_train, y_train), (X_test, y_test), (scaler_X, scaler_y), num_training_days, selected_columns = load_data()

# Define the autoencoder architecture
num_features = X_train.shape[1]

autoencoder, encoder = build_stacked_autoencoder((num_features,))

# Train the autoencoder
autoencoder.fit(X_train, y_train, epochs=400, batch_size=8, verbose=0, shuffle=True, validation_data=(X_test, y_test))

# Use encoder part of the autoencoder for feature selection
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

# Autocorrelation
def autocorrelation(data, lag=1):
    # Drop any columns that contain NaN values
    data_cleaned = data.dropna(axis=1)
    return [data_cleaned.iloc[:, i].autocorr(lag) for i in range(data_cleaned.shape[1])]

# Perform regression using Linear Regression
def perform_regression_test():
    regressor = LinearRegression()
    regressor.fit(encoded_features_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(encoded_features_test)

    # Inverse transform the predicted values and actual values
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate Mean Squared Error (MSE) on the original scale
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    print(f'Mean Squared Error (Original Scale): {mse_original}')

    return (y_test_original, y_pred_original), y_pred

# Plotting the results
def plot_result(predicted_data, real_data):
    plt.figure(figsize=(10, 5))
    plt.plot(real_data, label='True Values', color='blue')
    plt.plot(predicted_data, label='Predicted Values', color='orange', linestyle='--')
    plt.title('True vs Predicted Values (Original Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.show()

# (y_test_original, y_pred_original), y_pred = perform_regression_test()
# plot_result(y_pred_original, y_test_original)

# pip freeze > requirements.txt