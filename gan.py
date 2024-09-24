import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam
from autoencoder import encoded_features_train, y_train, target, encoded_features_test
from sklearn.metrics import mean_squared_error

num_training_days = int(target.shape[0]*.7)

# Define sequence length and number of features
time_step = 50
num_features = encoded_features_train.shape[1]

# Determine the number of complete sequences of length 50
num_samples = len(encoded_features_train) // time_step

# Truncate and reshape the data
data_train = encoded_features_train[:num_samples * time_step]
data_train = data_train.reshape(num_samples, time_step, num_features)
target_train = y_train[:num_samples * time_step]
target_train = target_train.reshape(num_samples, time_step, 1)

X = data_train[:, :-1, :]  # All but the last time step as input
y = target_train[:, -1, 0]   # Only price for the next time step

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(time_step - 1, num_features)))
    model.add(LSTM(400, return_sequences=True))
    model.add(LSTM(400))
    model.add(Dense(time_step - 1, activation='linear'))  # Change to linear activation
    model.add(Reshape((time_step - 1, 1)))  # Reshape to output only price
    return model

generator = build_generator()
generator.compile(loss='mean_squared_error', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(time_step - 1, 1)))  # Ensure this shape matches generated_data
    model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False  # Freeze discriminator when training the generator

def build_gan():
    gan_input = Input(shape=(time_step - 1, num_features))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    model = Model(gan_input, gan_output)
    return model

gan_model = build_gan()
gan_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, num_samples, batch_size)
        
        # Real data
        real_data = y[idx]
        real_data = np.repeat(real_data, time_step - 1)
        real_data = real_data.reshape(-1, time_step - 1, 1)

        # Generate synthetic data using the complete feature set
        generated_data = generator.predict(X[idx])
        
        # Ensure generated_data has the correct shape
        generated_data = generated_data.reshape(-1, time_step - 1, 1)  # Ensure shape matches

        print("Real data shape:", real_data.shape)
        print("Generated data shape:", generated_data.shape)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan_model.train_on_batch(X[idx], np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")


# Train the GAN
train_gan(epochs=50, batch_size=64)  # Increased epochs for better learning

# Generate New Price Series with context
def generate_new_series_with_context(last_data, num_samples):
    generated_series = []

    for _ in range(num_samples):
        context = last_data.reshape(1, time_step - 1, num_features)
        new_price = gan_model.predict(context)
        
        # Scale to [10, 300] if necessary (based on your original range)
        new_price_value = new_price[-1, 0, 0]
        new_price_value = np.clip(new_price_value, 0, 1)  # Clip to [0, 1]
        new_price_value = new_price_value * (300 - 10) + 10  # Scale to [10, 300]
        
        generated_series.append(new_price_value)

        # Update last_data with the new generated price and the context for other features
        new_features = last_data[1:].copy()  # Get the last features except price
        last_data = np.concatenate(([new_price_value], list(new_features)), axis=0)

    return np.array(generated_series)

# Use the last available sequence from the training data as context
target_test = target[num_training_days:]
last_history = encoded_features_test[-1,:]

# Generate new price series
generate_num = len(target_test)
new_data = generate_new_series_with_context(last_history, generate_num)

mse = mean_squared_error(target_test, new_data)
print("Mean Squared Error with Selected Features:", mse)

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(new_data, label='Generated', alpha=0.3)
plt.plot(target_test, label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
