import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, Dropout, TimeDistributed, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(time_step, num_features)))
    model.add(LSTM(32))
    model.add(Dense(64, activation='gelu'))
    model.add(Dense(128, activation='gelu'))
    model.add(Dense(256, activation='gelu'))
    model.add(Dense(time_step * num_features, activation='linear'))
    model.add(Reshape((time_step, num_features)))
    return model


generator = build_generator()
generator_optimizer = Adam(1e-4)

# Define the CNN Discriminator
def build_critic():
    model = Sequential()
    model.add(Input(shape=(time_step, num_features)))
    model.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation='leaky_relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

critic = build_critic()
critic_optimizer = Adam(1e-5)

# Build and compile the GAN
def build_gan():
    gan_input = Input(shape=(time_step, num_features))
    generated_data = generator(gan_input)
    gan_output = critic(generated_data)
    model = Model(gan_input, gan_output)
    return model

gan_model = build_gan()
gan_optimizer = Adam(1e-4)
gan_model.compile(loss='mean_squared_error', optimizer=gan_optimizer)

n_critic = 15  # Number of training steps for the critic per generator step
clip_value = 0.01

critic_losses = []
generator_losses = []

# Train the GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, num_samples, batch_size)
        
            # Real data
            real_data = X[idx].reshape(-1, time_step, num_features)

            # Generate synthetic data
            noise = np.random.normal(0, 1, (batch_size, time_step, num_features))
            fake_data = generator.predict(noise)

            with tf.GradientTape() as tape:
                real_loss = tf.reduce_mean(critic(real_data))
                fake_loss = tf.reduce_mean(critic(fake_data))
                d_loss = fake_loss - real_loss
            
            grads = tape.gradient(d_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            # Clip weights
            for weight in critic.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -clip_value, clip_value))

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, time_step, num_features))
        with tf.GradientTape() as tape:
            fake_data = generator(noise)
            g_loss = -tf.reduce_mean(critic(fake_data))

        grads = tape.gradient(g_loss, gan_model.trainable_variables)
        gan_optimizer.apply_gradients(zip(grads, gan_model.trainable_variables))

        critic_losses.append(d_loss.numpy())
        generator_losses.append(g_loss.numpy())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}')

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

batch_sizes = get_divisors(num_samples)
divider = len(batch_sizes) // 2
batch_size = batch_sizes[divider]

# Train the GAN
train_gan(epochs=150, batch_size=batch_size)

def plot_loss():
    plt.figure(figsize=(12, 6))
    plt.plot(critic_losses, label='Discriminator Loss', color='red')
    plt.plot(generator_losses, label='Generator Loss', color='blue')
    plt.title('Loss Trends Over Epochs')
    plt.xlabel('Epochs (every step)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss()

# Generate New Price Series with context
def generate_new_series_with_context(last_data, generate_num):
    generated_series = []

    for _ in range(generate_num):
        context = last_data.reshape(1, time_step, num_features)
        predicted_value = gan_model.predict(context)

        # Extract the new price
        predicted_price = predicted_value.reshape(1)[0]

        # Prepare features for the new history
        new_features = generator.predict(context)
        new_features = new_features[0, 0, :]

        # noise = np.random.uniform(10, 300, num_features)
        # new_features = scaler_X.transform(noise.reshape(1, -1)).flatten()

        new_features[0] = predicted_price  # Set the predicted price

        generated_series.append(predicted_price)

        # Update last_data with the new generated price and dynamic features
        last_data = last_data[1:]
        last_data = np.concatenate((last_data, [new_features]), axis=0)

    return np.array(generated_series)

# Evaluate the model using multiple metrics
def evaluate_model(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    print("Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

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
test_origin = scaler_y.inverse_transform(target_test.reshape(-1, 1)).flatten()

print('new_data', predict_origin[-10:])
print('test_origin', test_origin[-10:])

print('critic_losses', critic_losses[-1])
print('generator_losses', generator_losses[-1])

print('predict_origin.shape', predict_origin.shape)
print('test_origin.shape', test_origin.shape)

# Call the evaluation function after generating new data
evaluate_model(target_test, new_data)

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(new_data, label='Generated', alpha=0.3)
plt.plot(target_test, label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
