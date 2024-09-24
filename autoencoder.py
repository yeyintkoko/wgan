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
data = dataset.iloc[:, 1:].values
target = dataset.iloc[:, 0].values

# Standardize the training data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 50

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
print('y_train shape:', y_train.shape)


# Fit a LinearRegression regression model using the selected features
model = LinearRegression()
model.fit(encoded_features_train, y_train)

# Make predictions on the test set
y_pred = model.predict(encoded_features_test)

# Calculate accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error with Selected Features:", mse)