import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, MaxPooling1D, Dropout, TimeDistributed, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from autoencoder import encoded_features_test, encoded_features_train, y_train, y_test, scaler_y, scaler_X, num_training_days, get_divisors

# np.set_printoptions(suppress=True, precision=6)

features_train = encoded_features_train
features_test = encoded_features_test

target_train = y_train
target_test = y_test

# Learning rates
generator_lr = 1e-4
critic_lr = 1e-5

n_critic = 5  # Number of training steps for the critic per generator step
clip_value = 0.01

time_step = 50
num_epoch = 150

# Define sequence length and number of features
num_features = features_train.shape[1]
num_samples = len(features_train) - time_step

batch_sizes = get_divisors(num_samples)
# Adjust time_step size, so batch_size will be a divisor number of num_sample which is not 1 or the vlaue of num_samples
while len(batch_sizes) < 3:
    time_step = time_step - 1
    num_samples = len(features_train) - time_step
    batch_sizes = get_divisors(num_samples)

divider = len(batch_sizes) // 2
batch_size = batch_sizes[divider]

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
    model.add(Flatten())
    model.add(Dense(32, activation='gelu'))
    model.add(Dense(64, activation='gelu'))
    model.add(Dense(128, activation='gelu'))
    model.add(Dense(256, activation='gelu'))
    model.add(Dense(time_step * num_features, activation='linear'))
    model.add(Reshape((time_step, num_features)))
    return model

generator = build_generator()
# generator = load_model('generator_model.keras') # load trained generator
generator_optimizer = Adam(generator_lr)

# Define the CNN Discriminator
def build_critic():
    model = Sequential()
    model.add(Input(shape=(time_step, num_features)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

critic = build_critic()
# critic = load_model('critic_model.keras') # load trained critic
critic_optimizer = Adam(critic_lr)

# Build and compile the GAN
def build_gan():
    gan_input = Input(shape=(time_step, num_features))
    generated_data = generator(gan_input)
    gan_output = critic(generated_data)
    model = Model(gan_input, gan_output)
    return model

gan_model = build_gan()
# gan_model = load_model('gan_model.keras') # load trained gan_model
gan_optimizer = Adam(generator_lr)

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
            noise = tf.random.normal(shape=(batch_size, time_step, num_features))
            fake_data = generator(noise)

            with tf.GradientTape() as tape:
                real_loss = tf.reduce_mean(critic(real_data))
                fake_loss = tf.reduce_mean(critic(fake_data))
                c_loss = fake_loss - real_loss
            
            grads = tape.gradient(c_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            # Clip weights
            for weight in critic.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -clip_value, clip_value))

        # Train Generator
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(gan_model(noise))

        grads = tape.gradient(g_loss, gan_model.trainable_variables)
        gan_optimizer.apply_gradients(zip(grads, gan_model.trainable_variables))

        critic_losses.append(c_loss.numpy())
        generator_losses.append(g_loss.numpy())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Discriminator Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}')

# Train the GAN
train_gan(epochs=num_epoch, batch_size=batch_size)

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
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    print("Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

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
new_data = gan_model.predict(features_test_data).flatten()

# Inverse transform to get the actual predicted price
predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
test_origin = scaler_y.inverse_transform(target_test.reshape(-1, 1)).flatten()

generator.summary()
critic.summary()
gan_model.summary()

generator.save('generator_model.keras')
critic.save('critic_model.keras')
gan_model.save('gan_model.keras')

print('batch_sizes', batch_sizes)
print('batch_size', batch_size)
print('time_step', time_step)

print('new_data', new_data[-10:])
print('target_test', target_test[-10:])

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
