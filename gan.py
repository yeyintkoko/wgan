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

def get_hyperparams(time_step, features_train, reduce_index=0):
    num_samples = len(features_train) - time_step

    batch_sizes = get_divisors(num_samples)
    # Adjust time_step size, so batch_size will be a divisor number of num_sample which is not 1 or the vlaue of num_samples
    while len(batch_sizes) < 3:
        time_step = time_step - 1
        num_samples = len(features_train) - time_step
        batch_sizes = get_divisors(num_samples)

    divider = len(batch_sizes) // 2
    batch_size = batch_sizes[divider-reduce_index]
    return num_samples, time_step, batch_size, batch_sizes

# Truncate and reshape the data
def prepare_data(num_samples, time_step, features_train, target_train):
    samples = features_train[time_step:]
    targets = target_train[time_step:]
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
    return train_data, train_target

# Define the LSTM Generator
def build_generator(num_lstm, num_dense, time_step, num_features, num_base=16, num_hidden=32, dropout=0.2):
    input = Input(shape=(time_step, num_features))
    layer = input
    for i in range(num_lstm):
        multiplier = i + 1
        not_last_layer = multiplier != num_lstm
        layer = LSTM(num_hidden, return_sequences=not_last_layer, dropout=dropout, recurrent_dropout=dropout)(layer)
    if num_lstm < 1:
        layer = Flatten()(layer)
    for i in range(num_dense):
        multiplier = i + 1
        layer = Dense(num_base * (2**multiplier), activation='gelu')(layer)
    output = Dense(time_step * num_features, activation='linear')(layer)  # Output layer
    output = Reshape((time_step, num_features))(output)
    return Model(input, output)

# Define the CNN Discriminator
def build_critic(num_conv, num_dense, time_step, num_features, num_base_conv=16, num_base_dense=16):
    input = Input(shape=(time_step, num_features))
    layer = input
    for i in range(num_conv):
        multiplier = i + 1
        layer = Conv1D(num_base_conv * (2**multiplier), kernel_size=3, padding='same', activation='leaky_relu')(layer)
    # layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    for i in range(num_dense - 1, -1, -1):
        multiplier = i + 1
        layer = Dense(num_base_dense * (2**multiplier), activation='relu')(layer)
    output = Dense(1)(layer) # No activation for critic
    return Model(input, output)

# Build and compile the GAN
def build_gan(generator, critic, time_step, num_features):
    gan_input = Input(shape=(time_step, num_features))
    generated_data = generator(gan_input)
    gan_output = critic(generated_data)
    model = Model(gan_input, gan_output)
    return model

# Train the GAN
def train_gan(epochs, batch_size, X, y, num_samples, n_critic, clip_value, gan_lr, critic_lr, num_lstm, num_lstm_dense, num_lstm_hidden, num_lstm_base, dropout, num_conv, num_conv_dense, num_conv_base, num_conv_dense_base, time_step, num_features, patience=5, generator=None, critic=None, gan_model=None):
    if generator is None:
        generator = build_generator(num_lstm=num_lstm, num_dense=num_lstm_dense, time_step=time_step, num_features=num_features, num_hidden=num_lstm_hidden, num_base=num_lstm_base, dropout=dropout)
    
    if critic is None:
        critic = build_critic(num_conv=num_conv, num_dense=num_conv_dense, time_step=time_step, num_features=num_features, num_base_conv=num_conv_base, num_base_dense=num_conv_dense_base)
    critic_optimizer = Adam(critic_lr)

    if gan_model is None:
        gan_model = build_gan(generator, critic, time_step, num_features)
    gan_optimizer = Adam(gan_lr)

    critic_losses = []
    generator_losses = []

    # Initialize variables for early stopping and model checkpointing
    best_g_loss = float('inf')
    patience_counter = 0
    checkpoint_path = 'best_gan_model.keras'
    
    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, num_samples, batch_size)
        
            # Real data
            real_data = X[idx].reshape(-1, time_step, num_features)
            real_target = y[idx].reshape(-1, 1, 1)

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

            # Check for improvement
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                patience_counter = 0
                gan_model.save(checkpoint_path)  # Save the model
                print("Model improved and saved.")
            else:
                patience_counter += 1
            
            # Early stopping logic
            if patience_counter >= patience:
                print("Early stopping triggered.")
                return  # Exit the training loop
    return (gan_model, generator, critic), (critic_losses, generator_losses), best_g_loss

# Generate New Price Series with context
def generate_new_series_with_context(last_data, generate_num, time_step, num_features, gan_model):
    generated_series = []

    for _ in range(generate_num):
        context = last_data.reshape(1, time_step, num_features)
        prediction = gan_model.predict(context)
        predicted_value = prediction[0]

        # Extract the new price
        predicted_price = predicted_value.reshape(1)[0]

        # Prepare features for the new history
        new_features = prediction[1]

        # noise = np.random.uniform(10, 300, num_features)
        # new_features = scaler_X.transform(noise.reshape(1, -1)).flatten()

        new_features[0] = predicted_price  # Set the predicted price

        generated_series.append(predicted_price)

        # Update last_data with the new generated price and dynamic features
        last_data = last_data[1:]
        last_data = np.concatenate((last_data, new_features), axis=0)

    return np.array(generated_series)

def get_features_test_data(features_test, last_sample):
    features_test_data = []
    for i in range(len(features_test)):
        last_sample = last_sample[1:].copy()
        last_sample = np.concatenate((last_sample, [features_test[i, :]]), axis=0)
        features_test_data.append(last_sample)
    features_test_data = np.array(features_test_data)
    return features_test_data

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

def plot_loss(critic_losses, generator_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(critic_losses, label='Discriminator Loss', color='red')
    plt.plot(generator_losses, label='Generator Loss', color='blue')
    plt.title('Loss Trends Over Epochs')
    plt.xlabel('Epochs (every step)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

features_train = encoded_features_train
features_test = encoded_features_test

target_train = y_train
target_test = y_test

# Define sequence length and number of features
num_features = features_train.shape[1]

time_step = 50

reduce_index = 0
num_samples, time_step, batch_size, batch_sizes = get_hyperparams(time_step=time_step, features_train=features_train, reduce_index=reduce_index)

train_data, train_target = prepare_data(num_samples=num_samples, time_step=time_step, features_train=features_train, target_train=target_train)
X = train_data
y = train_target

# This block will only execute when this file is run directly
if __name__ == "__main__":
    # Learning rates
    gan_lr = 2e-4
    critic_lr = 1e-4

    n_critic = 6 # Number of training steps for the critic per generator step
    clip_value = 0.01
    patience = 50
    num_epoch = 150
    
    # LSTM
    num_lstm = 1
    num_lstm_hidden = 50

    num_lstm_dense = 3
    num_lstm_base = 32
    dropout = 0.2

    # Critic
    num_conv = 4
    num_conv_base = 32

    num_conv_dense = 3
    num_conv_dense_base = 32

    # Load trained models
    gan_model = None #load_model('best_gan_model.keras')
    generator = None #load_model('generator_model.keras')
    critic = None #load_model('critic_model.keras')

    # Train the GAN
    (gan_model, generator, critic), (critic_losses, generator_losses), best_g_loss = train_gan(epochs=num_epoch, batch_size=batch_size, X=X, y=y, num_samples=num_samples, n_critic=n_critic, clip_value=clip_value, gan_lr=gan_lr, critic_lr=critic_lr, num_lstm=num_lstm, num_lstm_dense=num_lstm_dense, num_lstm_hidden=num_lstm_hidden, num_lstm_base=num_lstm_base, dropout=dropout, num_conv=num_conv, num_conv_dense=num_conv_dense, num_conv_base=num_conv_base, num_conv_dense_base=num_conv_dense_base, time_step=time_step, num_features=num_features, patience=patience, generator=generator, critic=critic, gan_model=gan_model)
    

    # Generate new price series
    last_sample = train_data[-1]
    last_sample = last_sample[1:]
    last_history = np.concatenate((last_sample, [features_test[0, :]]), axis=0)

    generate_num = len(target_test)
    # new_data = generate_new_series_with_context(last_data=last_history, generate_num=generate_num, time_step=time_step, num_features=num_features, gan_model=gan_model)

    features_test_data = get_features_test_data(features_test, train_data[-1])
    new_data = gan_model.predict(features_test_data).flatten()

    # Inverse transform to get the actual predicted price
    predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
    test_origin = scaler_y.inverse_transform(target_test.reshape(-1, 1)).flatten()

    def save_models():
        generator.save('generator_model.keras')
        critic.save('critic_model.keras')
        gan_model.save('gan_model.keras')

    def plot_result():
        # Plot generated price series
        plt.figure(figsize=(10, 5))
        plt.plot(new_data, label='Generated', alpha=0.3)
        plt.plot(target_test, label='Real', alpha=0.7)  # Plot real prices
        plt.legend()
        plt.show()

    def visualize_result():
        generator.summary()
        critic.summary()
        gan_model.summary()

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
        # plot_loss(critic_losses, generator_losses)
        plot_result()

    save_models()
    visualize_result()

