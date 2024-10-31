import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import datetime

# Generate a sine wave
def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')

# Load dataset
dataset = pd.read_csv('data/output.csv', header=0)

# Define number of training days
num_training_days = int(dataset.shape[0] * 0.7)

train_data, test_data = dataset[:num_training_days], dataset[num_training_days:]

# Extract multiple features, assuming 'price' and 'MACD' are columns in the dataset
data = dataset[['price', 'ma7', '26ema', '12ema', 'upper_band', 'lower_band', 'ema', 'momentum']].values  # Adjust based on your actual features

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        y.append(data[i + time_step, 0])  # Predicting the price (first column)
    return np.array(X), np.array(y)

time_step = 10  # Number of time steps to look back
X, y = create_dataset(data, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Shape: (samples, time_step, features)

model = Sequential()
# shape=(time_step, features)
model.add(Input(shape=(time_step, data.shape[1])))  # Define the input shape using Input layer
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))  # Output layer for price prediction

model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X, y, epochs=500, batch_size=32, verbose=0)

# Starting point for the generation
last_sequence = data[-time_step:].reshape(1, time_step, data.shape[1])  # Shape: (1, time_step, num_features)

# Before the generation loop
last_known_volume = data[-1, 1]  # Get the last known volume from the original data

generated_data = []
num_generate = len(test_data)  # Number of new data points to generate

for _ in range(num_generate):
    # Get the next predicted value
    next_value = model.predict(last_sequence, verbose=0)  # Shape: (1, 1)
    
    print(f"Predicted next_value shape: {next_value.shape}, value: {next_value}")

    # Extract the scalar value
    next_value_scalar = next_value[0, 0]  # This should be a scalar
    
    # Append the predicted value to the generated data
    generated_data.append(next_value_scalar)

    # Prepare the next input sequence for the model
    # Here, you need to create a new sequence by appending the new value.
    next_value_reshaped = np.array([[next_value_scalar, last_sequence[0, -1, 1]]]).reshape(1, 1, data.shape[1])  # Replace the volume with the last known value or set a default

    # Update last_sequence by appending the new value
    last_sequence = np.append(last_sequence[:, 1:, :], next_value_reshaped, axis=1)
    
    print(f"Updated last_sequence shape: {last_sequence.shape}")

# Create an array for generated data with the same number of features
generated_data_full = np.array([[val, last_known_volume] for val in generated_data])
# Combine original and generated data for visualization
full_data = np.concatenate((data, generated_data_full), axis=0)


# Visualize the results
plt.plot(np.arange(len(data)), data[:, 0], label='Original Price Data')  # Plot original price
plt.plot(np.arange(len(data), len(full_data)), generated_data, label='Generated Price Data', color='red')
plt.title("Generated Time Series Data")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
