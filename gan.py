import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from autoencoder import encoded_features_test, encoded_features_train, y_train, y_test, scaler_y, scaler_X, num_training_days

# np.set_printoptions(suppress=True, precision=6)

features_train = encoded_features_train
features_test = encoded_features_test

target_train = y_train
target_test = y_test

# Define sequence length and number of features
time_step = 50
num_features = features_train.shape[1]
num_samples = len(features_train) // time_step

# Truncate and reshape the data
train_data = features_train[:num_samples * time_step]
train_data = train_data.reshape(num_samples, time_step, num_features)
train_target = target_train[:num_samples * time_step]
train_target = train_target.reshape(num_samples, time_step, 1)

X = train_data[:, :-1, :]
y = train_target[:, -1, 0]

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(time_step - 1, num_features)))
    model.add(LSTM(400, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.add(Reshape((1, 1)))
    return model

generator = build_generator()
generator.compile(loss='mean_squared_error', optimizer=Adam(0.0001, 0.5))

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(1, 1)))

    # Add the 1D Convolutional layers
    model.add(layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(layers.BatchNormalization(momentum=0.8))

    # Add the two Fully Connected layers
    model.add(layers.Flatten())  # Flatten the output before feeding into Dense layers
    model.add(layers.Dense(220, use_bias=False))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(0.01))
    model.add(layers.Dense(220, use_bias=False, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Binary classification
    return model

discriminator = build_discriminator()
discriminator.compile(loss='mean_squared_error', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

# Build and compile the GAN
def build_gan():
    gan_input = Input(shape=(time_step - 1, num_features))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    model = Model(gan_input, gan_output)
    return model

gan_model = build_gan()
gan_model.compile(loss='mean_squared_error', optimizer=Adam(0.0001, 0.5))

# Gradient penalty function
def gradient_penalty(real_data, fake_data):
    # Convert to TensorFlow tensors and ensure they are of the same type
    real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)
    fake_data = tf.convert_to_tensor(fake_data, dtype=tf.float32)

    # Randomly interpolate between real and fake data
    alpha = tf.random.uniform(shape=(real_data.shape[0], 1, 1), minval=0.0, maxval=1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Get the discriminator output for interpolated data
        d_interpolated = discriminator(interpolated)
    
    gradients = tape.gradient(d_interpolated, interpolated)
    gp = tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)
    return gp

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        idx = np.random.randint(0, num_samples, batch_size)
        
        # Real data
        real_data = y[idx]
        real_data = real_data.reshape(-1, 1, 1)

        # Generate synthetic data using the complete feature set
        noise = np.random.normal(0, 1, (batch_size, time_step - 1, num_features))
        fake_data = generator.predict(X[idx])
        
        # Calculate gradient penalty
        gp = gradient_penalty(real_data, fake_data)
        
        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) + 10 * gp  # Adding GP to the loss
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, time_step - 1, num_features))
        g_loss = gan_model.train_on_batch(X[idx], real_data)
        
        if epoch % 10 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

batch_sizes = get_divisors(num_samples)
divider = len(batch_sizes) // 2
batch_size = batch_sizes[divider]

# Train the GAN
train_gan(epochs=350, batch_size=batch_size)

def generate_new_feature(last_data, new_value):
    old_value = last_data[-1,0]
    old_features = last_data[-1]
    change_rate = ((new_value - old_value) / old_value)
    change_amounts = old_features * change_rate
    weighted_features = old_features + change_amounts
    weighted_features_scaled = scaler_X.transform(weighted_features.reshape(1, -1)).flatten()
    return weighted_features_scaled

# Generate New Price Series with context
def generate_new_series_with_context(last_data, generate_num):
    generated_series = []

    for _ in range(generate_num):
        context = last_data.reshape(1, time_step - 1, num_features)
        predicted_value = gan_model.predict(context)

        # Extract the new price
        predicted_price = predicted_value[0, 0]

        # Prepare features for the new history
        new_features = generate_new_feature(last_data, predicted_price)
        
        # noise = np.random.uniform(10, 300, num_features)
        # new_features = scaler_X.transform(noise.reshape(1, -1)).flatten()

        new_features[0] = predicted_price  # Set the predicted price

        generated_series.append(predicted_price)

        # Update last_data with the new generated price and dynamic features
        last_data = last_data[1:]
        last_data = np.concatenate((last_data, [new_features]), axis=0)

    return np.array(generated_series)

# Generate new price series
last_sample = train_data[-1]
last_sample = last_sample[2:]
last_history = np.concatenate((last_sample, [features_test[0, :]]), axis=0)

generate_num = len(target_test)
new_data = generate_new_series_with_context(last_history, generate_num)

# Inverse transform to get the actual predicted price
predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
test_origin = scaler_y.inverse_transform(target_test).flatten()

mse = mean_squared_error(target_test, new_data)
print("Mean Squared Error with Selected Features:", mse)
print('new_data', predict_origin[100:])

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(predict_origin, label='Generated', alpha=0.3)
plt.plot(test_origin, label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
