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
        layer = BatchNormalization()(layer)
    output = Dense(1, activation='linear')(layer)  # Output layer
    output = Reshape((1, 1))(output)
    return Model(input, output)

# Define the CNN Discriminator
def build_critic(num_conv, num_dense, time_step, num_features, num_base_conv=16, num_base_dense=16):
    input = Input(shape=(1, 1))
    layer = input
    for i in range(num_conv):
        multiplier = i + 1
        layer = Conv1D(num_base_conv * (2**multiplier), kernel_size=3, padding='same', activation='leaky_relu')(layer)
        layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    for i in range(num_dense - 1, -1, -1):
        multiplier = i + 1
        layer = Dense(num_base_dense * (2**multiplier), activation='relu')(layer)
        layer = BatchNormalization()(layer)
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
def train_gan(epochs, batch_size, X, y, num_samples, n_critic, clip_value, gan_lr, critic_lr, num_lstm, num_lstm_dense, num_lstm_hidden, num_lstm_base, dropout, num_conv, num_conv_dense, num_conv_base, num_conv_dense_base, time_step, num_features, patience=5, mape_patience=5, plot_epoch_interval=50, generator=None, critic=None, gan_model=None):
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

    best_mape = float('inf')
    mape_patience_counter = 0
    plot_epoch = False
    best_epoch = 0
    early_stop_triggered = False
    mape_patience_hitted = False

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, num_samples, batch_size)
        
            # Real data
            train_data = X[idx].reshape(-1, time_step, num_features)
            real_data = y[idx].reshape(-1, 1)

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
            g_loss = -tf.reduce_mean(gan_model(train_data))
            
        grads = tape.gradient(g_loss, generator.trainable_variables)
        gan_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        critic_losses.append(c_loss.numpy())
        generator_losses.append(g_loss.numpy())

        if epoch % plot_epoch_interval == 0:
            features_test_data = get_features_test_data(encoded_features_test, X[-1])
            new_data = generator.predict(features_test_data).flatten()
            mape = evaluate_model(target_test, new_data)
            if mape < 30:
                mape_patience_counter = 0
            else:
                plot_epoch = False

            if mape < 20:
                plot_epoch = True
                plot_epoch_interval = 50
            if plot_epoch:
                plot_result(new_data, target_test, epoch)

            if mape < best_mape:
                print(f"MAPE% improved: {best_mape - mape}")
                generator.save(checkpoint_path)  # Save the model
                best_mape = mape
                best_epoch = epoch
                mape_patience_counter = 0
            else:
                mape_patience_counter += 1
                
            # Early stopping logic: MAPE patience counter
            if mape_patience_counter >= mape_patience:
                mape_patience_hitted = True
                print(f'best_mape {best_mape} at epoch {best_epoch}')
                print("MAPE patience hit.")
                # Exit the training loop
                return (gan_model, generator, critic), (critic_losses, generator_losses), (best_g_loss, best_mape, best_epoch), (early_stop_triggered, mape_patience_hitted)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Discriminator Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}')

            # Check for improvement
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                patience_counter = 0
                generator.save(checkpoint_path)  # Save the model
                print("Model improved and saved.")
            else:
                patience_counter += 1
            
            # Early stopping logic
            if patience_counter >= patience:
                early_stop_triggered = True
                print("Early stopping triggered.")
                # Exit the training loop
                return (gan_model, generator, critic), (critic_losses, generator_losses), (best_g_loss, best_mape, best_epoch), (early_stop_triggered, mape_patience_hitted)

    return (gan_model, generator, critic), (critic_losses, generator_losses), (best_g_loss, best_mape, best_epoch), (early_stop_triggered, mape_patience_hitted)

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
    return mape

def plot_loss(critic_losses, generator_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(critic_losses, label='Discriminator Loss', color='red')
    plt.plot(generator_losses, label='Generator Loss', color='blue')
    plt.title('Loss Trends Over Epochs')
    plt.xlabel('Epochs (every step)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_train(features_train, target_train):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(features_train)
    plt.title('Train Data')

    plt.subplot(2, 1, 2)
    plt.plot(target_train)
    plt.title('Train Target')

    plt.show()

def plot_test(features_test, target_test):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(features_test)
    plt.title('Test Data')

    plt.subplot(2, 1, 2)
    plt.plot(target_test)
    plt.title('Test Target')

    plt.show()

def plot_result(generated_data, real_data, epoch=None):
    # Plot generated price series
    plt.figure(figsize=(10, 5))
    plt.plot(generated_data, label='Generated', alpha=0.3)
    plt.plot(real_data, label='Real', alpha=0.7)  # Plot real prices
    if epoch:
        plt.title(f'Epoch {epoch}')
    plt.legend()
    plt.show()

# Seed for reproducibility
np.random.seed(42)

features_train = encoded_features_train
features_test = encoded_features_test

target_train = y_train
target_test = y_test

# Define sequence length and number of features
num_features = features_train.shape[1]

time_step = 150

reduce_index = 0
num_samples, time_step, batch_size, batch_sizes = get_hyperparams(time_step=time_step, features_train=features_train, reduce_index=reduce_index)

train_data, train_target = prepare_data(num_samples=num_samples, time_step=time_step, features_train=features_train, target_train=target_train)
X = train_data
y = train_target

# This block will only execute when this file is run directly
if __name__ == "__main__":
    # Learning rates
    gan_lr = 1e-5
    critic_lr = 1e-6

    n_critic = 5 # Number of training steps for the critic per generator step
    clip_value = 0.01

    patience = 100
    mape_patience = 3
    plot_epoch_interval = 150
    num_epoch = 6500
    
    # LSTM
    num_lstm = 0
    num_lstm_hidden = 16

    num_lstm_dense = 2
    num_lstm_base = 64
    dropout = 0.2

    # Critic
    num_conv = 2
    num_conv_base = 64

    num_conv_dense = 0
    num_conv_dense_base = 64

    # Load trained models
    gan_model = None #load_model('best_gan_model.keras')
    generator = None #load_model('generator_model.keras')
    critic = None #load_model('critic_model.keras')
    
    # plot_train(features_train, target_train)

    def automate_train():
        models, losses, bests, breaks = train_gan(epochs=num_epoch, batch_size=batch_size, X=X, y=y, num_samples=num_samples, n_critic=n_critic, clip_value=clip_value, gan_lr=gan_lr, critic_lr=critic_lr, num_lstm=num_lstm, num_lstm_dense=num_lstm_dense, num_lstm_hidden=num_lstm_hidden, num_lstm_base=num_lstm_base, dropout=dropout, num_conv=num_conv, num_conv_dense=num_conv_dense, num_conv_base=num_conv_base, num_conv_dense_base=num_conv_dense_base, time_step=time_step, num_features=num_features, patience=patience, mape_patience=mape_patience, plot_epoch_interval=plot_epoch_interval, generator=generator, critic=critic, gan_model=gan_model)
        (early_stop_triggered, mape_patience_hitted) = breaks
        if early_stop_triggered:
            print('💥💣🧨🔥 early_stop_triggered 🔥🧨💣💥')
        if mape_patience_hitted:
            print('💥💣🧨🔥 mape_patience_hitted 🔥🧨💣💥')
        if early_stop_triggered or mape_patience_hitted:
            automate_train()
        return models, losses, bests, breaks
    
    # Train the GAN
    (gan_model, generator, critic), (critic_losses, generator_losses), (best_g_loss, best_mape, best_epoch), (early_stop_triggered, mape_patience_hitted) = automate_train()
    
    # Generate new price series
    last_sample = train_data[-1]
    last_sample = last_sample[1:]
    last_history = np.concatenate((last_sample, [features_test[0, :]]), axis=0)

    generate_num = len(target_test)
    # new_data = generate_new_series_with_context(last_data=last_history, generate_num=generate_num, time_step=time_step, num_features=num_features, gan_model=gan_model)

    features_test_data = get_features_test_data(features_test, train_data[-1])
    new_data = generator.predict(features_test_data).flatten()

    # Inverse transform to get the actual predicted price
    predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
    test_origin = scaler_y.inverse_transform(target_test.reshape(-1, 1)).flatten()

    def save_models():
        generator.save('generator_model.keras')
        critic.save('critic_model.keras')
        gan_model.save('gan_model.keras')

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

        print('features_train.shape', features_train.shape)
        print('target_train.shape', target_train.shape)

        print('new_data.shape', new_data.shape)
        print('target_test.shape', target_test.shape)

        print('predict_origin.shape', predict_origin.shape)
        print('test_origin.shape', test_origin.shape)

        print(f'best_mape {best_mape} at epoch {best_epoch}')

        # Call the evaluation function after generating new data
        evaluate_model(target_test, new_data)

        # plot_train(features_train, target_train)
        # plot_test(features_test, target_test)
        # plot_loss(critic_losses, generator_losses)
        plot_result(new_data, target_test)

    save_models()
    visualize_result()

