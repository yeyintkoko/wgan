import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Generate a sine wave
def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')

dataset = pd.read_csv('data/panel_data_close.csv', header=0, parse_dates=[0], date_parser=parser)
num_training_days = int(dataset.shape[0]*.7)

train_data, test_data = dataset[:num_training_days], dataset[num_training_days:]

time = np.array(dataset['Date'])
data = np.array(dataset['price'])

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 10  # Number of time steps to look back
X, y = create_dataset(data, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=100, batch_size=32)

# Starting point for the generation
last_sequence = data[-time_step:].reshape(1, time_step, 1)

generated_data = []
num_generate = len(test_data)  # Number of new data points to generate

for _ in range(num_generate):
    # Get the next predicted value
    next_value = model.predict(last_sequence)  # Shape: (1, 1)
    
    print(f"Predicted next_value shape: {next_value.shape}, value: {next_value}")

    # Extract the scalar value
    next_value_scalar = next_value[0, 0]  # This should be a scalar
    
    # Append the predicted value to the generated data
    generated_data.append(next_value_scalar)

    # Reshape next_value_scalar to (1, 1, 1)
    next_value_reshaped = np.array(next_value_scalar).reshape(1, 1, 1)  # Shape: (1, 1, 1)
    
    print(f"next_value_reshaped shape: {next_value_reshaped.shape}")

    # Update last_sequence by appending the new value
    last_sequence = np.append(last_sequence[:, 1:, :], next_value_reshaped, axis=1)
    
    print(f"Updated last_sequence shape: {last_sequence.shape}")

# Combine original and generated data for visualization
full_data = np.concatenate((data, generated_data))

# Visualize the results
plt.plot(np.arange(len(data)), data, label='Original Data')
plt.plot(np.arange(len(data), len(full_data)), generated_data, label='Generated Data', color='red')
plt.title("Generated Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

