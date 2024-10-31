import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

dataset = pd.read_csv("data/output.csv", header=0).dropna()
dataset = dataset[['price', 'ma7', '26ema', '12ema', 'upper_band', 'lower_band', 'ema', 'momentum']]
dataset = dataset[::-1].reset_index(drop=True)

# Use the first feature as the target
features = np.array(dataset.iloc[:, :])
target = np.array(dataset.iloc[:, 0])

num_features = features.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, shuffle=False)

# Reshape the data for LSTM input
X_train = X_train.reshape(-1, 1, num_features)  # (samples, timesteps, features)
X_test = X_test.reshape(-1, 1, num_features)

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(1, num_features)),
        layers.LSTM(32, activation='relu'),
        layers.Dense(1)  # Output 1D array
    ])
    return model

# Critic
def build_critic():
    model = tf.keras.Sequential([
        layers.Conv1D(64, kernel_size=2, strides=1, padding='same', input_shape=(1, num_features)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, kernel_size=2, strides=1, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
critic = build_critic()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)

# Compile the models with optimizers
generator.compile(optimizer=generator_optimizer, loss='mean_squared_error')
critic.compile(optimizer=critic_optimizer, loss='mean_squared_error')

def train_wgan(generator, critic, X_train, y_train, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Train the critic
        for _ in range(5):  # Train critic more than generator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real = y_train[idx]

            # Generate fake data
            noise = generator.predict(X_train[idx], verbose=0)
            
            # Train the critic with real and fake data
            critic_loss_real = critic.train_on_batch(X_train[idx], real)
            critic_loss_fake = critic.train_on_batch(X_train[idx], noise)

            # Calculate total critic loss
            critic_loss = critic_loss_real - critic_loss_fake

        # Train the generator
        noise = generator.predict(X_train[np.random.randint(0, X_train.shape[0], batch_size)], verbose=0)
        generator_loss = -critic.train_on_batch(X_train[np.random.randint(0, X_train.shape[0], batch_size)], noise)

        # Print losses
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Critic Loss: {critic_loss}, Generator Loss: {generator_loss}')

# Train the WGAN
train_wgan(generator, critic, X_train, y_train)

# Generate predictions on the test set
predictions = generator.predict(X_test).flatten()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', linestyle='dashed')
plt.title('WGAN Predictions vs Actual Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
