import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam

# Generate synthetic time series data
def generate_time_series_data(n_samples=1000, timesteps=50):
    x = np.linspace(0, 100, timesteps)
    data = np.sin(x) + np.random.normal(0, 0.1, (n_samples, timesteps))  # Sinusoidal data with noise
    return data

data = generate_time_series_data()
data = (data - np.mean(data)) / np.std(data)  # Normalize
data = data.reshape(-1, 50, 1)  # Reshape for LSTM

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(LSTM(50, input_shape=(50, 1), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='tanh'))
    model.add(Reshape((50, 1)))  # Reshape back to time-series format
    return model

generator = build_generator()

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, strides=2, padding='same', input_shape=(50, 1)))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding='same'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

discriminator = build_discriminator()

# Compile the Discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Note: Compile the Generator (not required for GAN training)


# Build the GAN
discriminator.trainable = False  # Freeze discriminator when training the generator

gan_input = Sequential()
gan_input.add(generator)
gan_input.add(discriminator)

gan_input.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, 50, 1))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 50, 1))
        g_loss = gan_input.train_on_batch(noise, np.ones((batch_size, 1)))[0]  # Access the first element (loss)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

# Train the GAN
train_gan(epochs=10000, batch_size=32)

# Generate New Time Series
def generate_new_series(num_samples):
    noise = np.random.normal(0, 1, (num_samples, 50, 1))
    generated_series = generator.predict(noise)
    return generated_series

new_data = generate_new_series(5)

# Plot generated series
for i in range(new_data.shape[0]):
    plt.plot(new_data[i].reshape(50,), label=f"Generated Series {i+1}")
plt.legend()
plt.show()
