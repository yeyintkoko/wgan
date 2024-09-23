import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam

# Load dataset
dataset = pd.read_csv('data/output.csv', header=0)

# Define number of training days
num_training_days = int(dataset.shape[0] * 0.7)
train_data, test_data = dataset[:num_training_days], dataset[num_training_days:]

# Extract features, using 'price' for output and all available features for input
data = train_data.dropna().values  # Use all columns in the dataset

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Define sequence length and number of features
time_step = 50
num_features = data.shape[1]

# Determine the number of complete sequences of length 50
num_samples = len(data) // time_step

# Truncate and reshape the data
data = data[:num_samples * time_step]
data = data.reshape(num_samples, time_step, num_features)

# Prepare input X (all features) and output y (only price)
X = data[:, :-1, :]  # All but the last time step as input
y = data[:, -1, 0]   # Only price for the next time step

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(LSTM(400, input_shape=(time_step - 1, num_features), return_sequences=True))
    model.add(LSTM(400))
    model.add(Dense(num_features, activation='linear'))  # Change to linear activation
    model.add(Reshape((1, num_features)))  # Reshape to output only price
    return model

generator = build_generator()

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    # Add the two Fully Connected layers
    model.add(layers.Flatten())  # Flatten the output before feeding into Dense layers
    model.add(layers.Dense(220, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.01))
    model.add(layers.Dense(220, use_bias=False, activation='relu'))
    model.add(layers.Dense(1))
    return model

discriminator = build_discriminator()
discriminator.compile(loss='mean_squared_error', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False  # Freeze discriminator when training the generator

gan_input = Sequential()
gan_input.add(generator)
gan_input.add(discriminator)

gan_input.compile(loss='mean_squared_error', optimizer=Adam(0.0002, 0.5))

def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
    return data + noise

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, num_samples, batch_size)
        
        # Real data
        real_data = add_noise(y[idx].reshape(-1, 1))  # Shape: (batch_size, 1)
        
        # Check and repeat to create 20 features
        real_data = np.repeat(real_data, 20, axis=1)  # Now shape is (batch_size, 20)
        real_data = real_data.reshape(batch_size, 1, 20)  # Reshape to (batch_size, 1, 20)

        # Generate synthetic data using the complete feature set
        generated_data = generator.predict(X[idx])
        
        # Ensure generated_data has the correct shape
        generated_data = generated_data.reshape(batch_size, 1, 20)  # Ensure shape matches

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan_input.train_on_batch(X[idx], np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")


# Train the GAN
train_gan(epochs=150, batch_size=64)  # Increased epochs for better learning

# Generate New Price Series with context
def generate_new_series_with_context(last_data, num_samples):
    generated_series = []

    for _ in range(num_samples):
        context = last_data[-(time_step - 1):].reshape(1, time_step - 1, num_features)
        new_price = generator.predict(context)
        
        # Scale to [10, 600] if necessary (based on your original range)
        new_price_value = new_price[-1, 0, 0]
        new_price_value = np.clip(new_price_value, 0, 1)  # Clip to [0, 1]
        new_price_value = new_price_value * (600 - 10) + 10  # Scale to [10, 600]
        
        generated_series.append(new_price_value)

        # Update last_data with the new generated price and the context for other features
        new_features = last_data[-1, 1:].copy()  # Get the last features except price
        last_data = np.concatenate((last_data[1:], [[new_price_value] + list(new_features)]), axis=0)

    return np.array(generated_series)

# Use the last available sequence from the training data as context
last_context = data[-1]

# Generate new price series
generate_num = len(test_data)
new_data = generate_new_series_with_context(last_context, generate_num)

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(new_data, label='Generated', alpha=0.3)
plt.plot(test_data.values[:, 0], label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
