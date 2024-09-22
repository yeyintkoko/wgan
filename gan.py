import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam

# Load dataset
dataset = pd.read_csv('data/output.csv', header=0)

# Define number of training days
num_training_days = int(dataset.shape[0] * 0.7)
train_data, test_data = dataset[:num_training_days], dataset[num_training_days:]

# Extract features, using 'price' for output and 'price' & 'ma7' for input
data = train_data[['price', 'momentum', '26ema']].values

# Normalize the data (optional based on your data)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Define sequence length and number of features
time_step = 50
num_features = data.shape[1]

# Determine the number of complete sequences of length 50
num_samples = len(data) // time_step

# Truncate and reshape the data
data = data[:num_samples * time_step]
data = data.reshape(num_samples, time_step, num_features)

# Prepare input X (using both features) and output y (only price)
X = data[:, :-1, :]
y = data[:, 1:, 0]

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(LSTM(50, input_shape=(time_step - 1, num_features), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(time_step - 1, activation='linear'))  # Change to linear activation
    model.add(Reshape((time_step - 1, 1)))  # Reshape to output only price
    return model

generator = build_generator()

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, strides=2, padding='same', input_shape=(time_step - 1, 1)))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding='same'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False  # Freeze discriminator when training the generator

gan_input = Sequential()
gan_input.add(generator)
gan_input.add(discriminator)

gan_input.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, num_samples, batch_size)
        real_data = y[idx].reshape(-1, time_step - 1, 1)

        # Generate synthetic data using X
        noise = np.random.normal(50, 300, (batch_size, time_step - 1, num_features))
        generated_data = generator.predict(X[idx])
        
        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan_input.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")

# Train the GAN
train_gan(epochs=50, batch_size=32)  # Increase epochs for better learning

# Generate New Price Series with context
def generate_new_series_with_context(last_data, num_samples):
    generated_series = []

    for _ in range(num_samples):
        context = last_data[-(time_step - 1):].reshape(1, time_step - 1, num_features)
        new_price = generator.predict(context)
        # Scale to [50, 300]
        # new_price_value = new_price[-1, 0, 0]  # Get the generated price
        # new_price_value = np.clip(new_price_value, 0, 1)  # Clip to [0, 1] to avoid out-of-bounds values
        # new_price_value = new_price_value * (300 - 50) + 50
        generated_series.append(new_price)

        # Extract the last generated price and create a placeholder for ma7
        new_price_value = new_price[-1, 0, 0]  # Extract the last price from the 3D array
        new_ma7_value = 0  # Placeholder value for ma7
        last_data = np.concatenate((last_data[1:], [[new_price_value, 0, 0]]), axis=0)
        # last_data = np.concatenate((last_data[1:], [[new_price_value, new_ma7_value]]), axis=0)  # Update last_data with the new price and MACD

    return np.array(generated_series)

# Use the last available sequence from the training data as context
last_context = data[-1]

# Generate new price series
generate_num = len(test_data) // time_step
new_data = generate_new_series_with_context(last_context, generate_num)

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(new_data.reshape(-1), label='Generated Price', alpha=0.3)
# plt.plot(test_data.values[:, 0], label='Real Price', alpha=0.7)  # Plot real prices
plt.title("Generated Price Series vs Real Price")
plt.legend()
plt.show()
