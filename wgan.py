# Wasserstein GAN (WGAN)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # Shape (num_samples, 28, 28, 1)

# WGAN parameters
latent_dim = 100
batch_size = 64
num_epochs = 10000
n_critic = 5  # Number of training steps for the critic per generator step
clip_value = 0.01  # Weight clipping parameter

# Build the generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=latent_dim),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Build the critic model
def build_critic():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)  # Output a single value
    ])
    return model

# Create models
generator = build_generator()
critic = build_critic()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
critic_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
for epoch in range(num_epochs):
    for _ in range(n_critic):
        # Train the critic
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        with tf.GradientTape() as tape:
            real_loss = tf.reduce_mean(critic(real_imgs))
            fake_loss = tf.reduce_mean(critic(fake_imgs))
            critic_loss = fake_loss - real_loss

        grads = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

        # Clip weights
        for weight in critic.trainable_variables:
            weight.assign(tf.clip_by_value(weight, -clip_value, clip_value))

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    with tf.GradientTape() as tape:
        fake_imgs = generator(noise)
        generator_loss = -tf.reduce_mean(critic(fake_imgs))

    grads = tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # Log the progress
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Critic Loss: {critic_loss.numpy()}, Generator Loss: {generator_loss.numpy()}')
        
        # Generate some images to visualize
        generated_images = generator.predict(np.random.normal(0, 1, (16, latent_dim)))
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
