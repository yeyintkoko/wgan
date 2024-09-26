# Import necessary libraries for loading and processing data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Import Keras for implementing autoencoders
import keras 
from keras.models import Sequential

# Load the dataset
dataset = pd.read_csv("data/output.csv", header=0).dropna()
dataset = dataset[::-1].reset_index(drop=True)
train_data = dataset.iloc[:, 1:].values
target_data = dataset.iloc[:, 0].values

# Set print options for NumPy
np.set_printoptions(suppress=True, precision=6)

# Standardize the training data
scaler = StandardScaler()
data = scaler.fit_transform(train_data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target_data, test_size=0.3, random_state=42)

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 25

# Set the encoding dimension
input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Summary of the autoencoder architecture
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Use encoder part of the autoencoder for feature selection
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

# Display the shape of extracted features
print("Encoded Features Shape (Train):", encoded_features_train.shape)
print("Encoded Features Shape (Test):", encoded_features_test.shape)

# ------------------ autocorrelation ------------

def autocorrelation(data, lag=1):
    """Calculate autocorrelation for each feature."""
    return [data.iloc[:, i].autocorr(lag) for i in range(data.shape[1])]

autocorr_values = autocorrelation(pd.DataFrame(encoded_features_train), lag=1)

# Convert autocorr_values to a NumPy array for easy manipulation
weights = np.array(autocorr_values)

# Apply the weights to your input features
weighted_encoded_features_train = encoded_features_train * weights