import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models

# 1. Generate synthetic time series data
def generate_time_series_data(n_samples=1000):
    time = np.arange(n_samples)
    data = np.sin(0.1 * time) + 0.1 * np.random.normal(size=n_samples)  # Sine wave with noise
    return data

data = generate_time_series_data()
data = pd.DataFrame(data)

# 2. Build the Stacked Autoencoder
def build_stacked_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(32, activation='relu')(encoded)
    
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(decoded)
    
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

input_shape = (data.shape[1],)
autoencoder, encoder = build_stacked_autoencoder(input_shape)

# 3. Train the autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=32, shuffle=True)

# 4. Extract features using the encoder
encoded_data = encoder.predict(data)

# 5. Build the GAN
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(32, activation='linear'))  # Update output shape to match encoded features
    return model

def build_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_shape[0]))  # Change input_dim to match encoded data shape
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Build the GAN components
latent_dim = 32  # This should match the size of the encoded features
generator = build_generator(latent_dim)
discriminator = build_discriminator((32,))  # Update this to match the encoded data shape

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
fake_time_series = generator(gan_input)
gan_output = discriminator(fake_time_series)
gan = models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# 6. Train the GAN
def train_gan(epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, encoded_data.shape[0], batch_size)
        real_data = encoded_data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}] [G loss: {g_loss[0]:.4f}]")

# Start training the GAN
train_gan(epochs=50)

# 7. Generate new time series data
noise = np.random.normal(0, 1, (100, latent_dim))
generated_data = generator.predict(noise)

# Plot generated time series
plt.plot(generated_data)
plt.title('Generated Time Series')
plt.show()