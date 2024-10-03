import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, Dropout, GRU, LeakyReLU
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
num_samples = len(features_train) - time_step

samples = features_train[time_step:]
targets = target_train[time_step:]

# Truncate and reshape the data
last_sample = features_train[:time_step]
train_data = []
train_target = []
for i in range(num_samples):
    current_target = targets[i]
    current_train = samples[i]
    
    train_data.append(last_sample)
    train_target.append(current_target)
    
    last_sample = last_sample[1:]
    last_sample = np.concatenate((last_sample, [current_train]))

train_data = np.array(train_data)
train_target = np.array(train_target)

X = train_data
y = train_target

# Define the LSTM/GRU Generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(time_step, num_features)))
    model.add(Flatten())
    layers.Dense(256, activation='gelu'),
    layers.Dense(512, activation='gelu'),
    layers.Dense(1024, activation='gelu'),
    model.add(Dense(num_features, activation='linear'))
    model.add(Reshape((1, num_features)))
    return model

generator = build_generator()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(1, num_features)))
    model.add(Flatten())
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    model.add(Dense(1, activation='linear'))
    return model

discriminator = build_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Build and compile the GAN
def build_gan():
    gan_input = Input(shape=(time_step, num_features))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    model = Model(gan_input, gan_output)
    return model

gan_model = build_gan()
gan_optimizer = tf.keras.optimizers.Adam(1e-4)
gan_model.compile(loss='mean_squared_error', optimizer=gan_optimizer)

n_discriminator = 5  # Number of training steps for the discriminator per generator step
clip_value = 0.01

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(n_discriminator):
            idx = np.random.randint(0, num_samples, batch_size)
        
            # Real data
            real_data = X[idx]
            real_data = real_data.reshape(-1, 1, num_features)

            # Generate synthetic data using the complete feature set
            noise = np.random.normal(0, 1, (batch_size, time_step, num_features))
            fake_data = generator.predict(X[idx])

            with tf.GradientTape() as tape:
                real_loss = tf.reduce_mean(discriminator(real_data))
                fake_loss = tf.reduce_mean(discriminator(fake_data))
                d_loss = fake_loss - real_loss
            
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            # Clip weights
            for weight in discriminator.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -clip_value, clip_value))
            
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, time_step, num_features))
        with tf.GradientTape() as tape:
            fake_data = generator(noise)
            g_loss = -tf.reduce_mean(discriminator(fake_data))

        grads = tape.gradient(g_loss, gan_model.trainable_variables)
        gan_optimizer.apply_gradients(zip(grads, gan_model.trainable_variables))

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}')

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

batch_sizes = get_divisors(num_samples)
divider = len(batch_sizes) // 2
batch_size = batch_sizes[divider]

# Train the GAN
train_gan(epochs=250, batch_size=batch_size)

def generate_new_feature(last_data, new_value):
    old_value = last_data[-1,0]
    old_features = last_data[-1]
    change_rate = ((new_value - old_value) / old_value)
    change_rate = scaler_y.transform(np.array(change_rate).reshape(-1,1)).flatten()
    change_amounts = old_features * change_rate
    weighted_features = old_features + change_amounts
    weighted_features_scaled = scaler_X.transform(weighted_features.reshape(1, -1)).flatten()
    return weighted_features_scaled

# Generate New Price Series with context
def generate_new_series_with_context(last_data, generate_num):
    generated_series = []

    for _ in range(generate_num):
        context = last_data.reshape(1, time_step, num_features)
        predicted_value = gan_model.predict(context)

        # Extract the new price
        predicted_price = predicted_value.reshape(1)[0]

        new_features = generator.predict(context)
        new_features = new_features.flatten()

        # Prepare features for the new history
        # new_features = generate_new_feature(last_data, predicted_price)
        
        # noise = np.random.uniform(10, 300, num_features)
        # new_features = scaler_X.transform(noise.reshape(1, -1)).flatten()

        new_features[0] = predicted_price  # Set the predicted price
        print('------ new_features -----', new_features)

        generated_series.append(predicted_price)

        # Update last_data with the new generated price and dynamic features
        last_data = last_data[1:]
        last_data = np.concatenate((last_data, [new_features]), axis=0)

    return np.array(generated_series)

# Generate new price series
last_sample = train_data[-1]
last_sample = last_sample[1:]
last_history = np.concatenate((last_sample, [features_test[0, :]]), axis=0)

generate_num = len(target_test)
# new_data = generate_new_series_with_context(last_history, generate_num)

last_sample_2 = train_data[-1]
features_test_data = []
for i in range(len(features_test)):
    last_sample_2 = last_sample_2[1:].copy()
    last_sample_2 = np.concatenate((last_sample_2, [features_test[i, :]]), axis=0)
    features_test_data.append(last_sample_2)
features_test_data = np.array(features_test_data)
new_data = gan_model.predict(features_test_data)

# Inverse transform to get the actual predicted price
predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
test_origin = scaler_y.inverse_transform(target_test).flatten()

mse = mean_squared_error(target_test, new_data)
print("Mean Squared Error with Selected Features:", mse)
# print('new_data', predict_origin[100:])

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(predict_origin, label='Generated', alpha=0.3)
plt.plot(test_origin, label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
