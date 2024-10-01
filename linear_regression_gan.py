import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 1) * 10  # 1000 samples, single feature
Y = 2.5 * X + 1.0 + np.random.normal(0, 1, X.shape)  # Linear relation with noise

# Split into train set
X_train = X[:800]
Y_train = Y[:800]

# Define the generator
def build_generator():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_dim=1),
        layers.Dense(1)  # Output layer to predict Y
    ])
    return model

# Define the discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_dim=1),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    return model

# Build and compile the GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combine GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(1,))
generated_output = generator(gan_input)
gan_output = discriminator(generated_output)

gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training the GAN
epochs = 10000
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], 32)
    real_X = X_train[idx]
    real_Y = Y_train[idx]

    fake_Y = generator.predict(real_X)
    d_loss_real = discriminator.train_on_batch(real_Y, np.ones((32, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_Y, np.zeros((32, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.rand(32, 1) * 10  # Input noise
    g_loss = gan.train_on_batch(noise, np.ones((32, 1)))  # We want to fool the discriminator

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Evaluate the generator
import matplotlib.pyplot as plt

predictions = generator.predict(X)

plt.scatter(X, Y, label='Real Data', alpha=0.5)
plt.scatter(X, predictions, label='GAN Predictions', alpha=0.5)
plt.legend()
plt.show()
