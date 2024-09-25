import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, Conv1D, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from autoencoder import encoded_features_test, weighted_encoded_features_train, y_train, target, scaler, data

np.set_printoptions(suppress=True, precision=6)

num_training_days = int(target.shape[0] * .7)

# Define sequence length and number of features
time_step = 50
num_features = weighted_encoded_features_train.shape[1]
num_samples = len(weighted_encoded_features_train) // time_step

# Truncate and reshape the data
data_train = weighted_encoded_features_train[:num_samples * time_step]
data_train = data_train.reshape(num_samples, time_step, num_features)
target_train = y_train[:num_samples * time_step]
target_train = target_train.reshape(num_samples, time_step, 1)

X = data_train[:, :-1, :]
y = target_train[:, -1, 0]

# Define the LSTM Generator
def build_generator():
    model = Sequential()
    model.add(Input(shape=(time_step - 1, num_features)))
    model.add(LSTM(400, return_sequences=True))
    model.add(LSTM(400))
    model.add(Dense(1, activation='linear'))
    model.add(Reshape((1, 1)))
    return model

generator = build_generator()
generator.compile(loss='mean_squared_error', optimizer=Adam(0.0002, 0.5))

# Define the CNN Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(1, 1)))

    # Add the 1D Convolutional layers
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
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the GAN
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
        idx = np.random.randint(0, num_samples, batch_size)
        
        # Real data
        real_data = y[idx]
        real_data = real_data.reshape(-1, 1, 1)  # Ensure the shape matches

        # Generate synthetic data using the complete feature set
        generated_data = generator.predict(X[idx])
        
        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan_model.train_on_batch(X[idx], real_data)
        
        if epoch % 10 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")
        

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

batch_sizes = get_divisors(num_samples)
divider = len(batch_sizes) // 2
batch_size = batch_sizes[divider]

# Train the GAN
train_gan(epochs=50, batch_size=batch_size)

# Generate New Price Series with context
def generate_new_series_with_context(last_data, num_samples):
    generated_series = []
    last_data = np.repeat(last_data, time_step - 1)

    for _ in range(num_samples):
        context = last_data.reshape(1, time_step - 1, num_features)
        new_price = gan_model.predict(context)

        new_price = new_price[0, 0]  # Extract the new price
        num_actual_feature = data.shape[1]
        new_price_features = np.array([[new_price] + [0] * (num_actual_feature - 1)]) # 0 for the features
        new_price_values = scaler.inverse_transform(new_price_features) # reverse scale transform to actual value
        predicted_price = new_price_values[0,0]
        
        generated_series.append(predicted_price)

        # Update last_data with the new generated price
        new_features = last_data[1:].copy()  # Get the last features except price
        last_data = np.concatenate(([predicted_price], new_features), axis=0)

    return np.array(generated_series)

# Generate new price series
last_history = encoded_features_test[0, :]
generate_num = len(target[num_training_days:])
new_data = generate_new_series_with_context(last_history, generate_num)

mse = mean_squared_error(target[num_training_days:], new_data)
print("Mean Squared Error with Selected Features:", mse)

# Plot generated price series
plt.figure(figsize=(10, 5))
plt.plot(new_data, label='Generated', alpha=0.3)
plt.plot(target[num_training_days:], label='Real', alpha=0.7)  # Plot real prices
plt.title("Generated Series vs Real")
plt.legend()
plt.show()
