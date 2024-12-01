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

# Define the CNN Generator
def build_generator(num_lstm, base_lstm, num_dense, base_dense, time_step, num_features):
    input = Input(shape=(time_step, num_features))
    layer = input
    for i in range(num_lstm):
        not_last_layer = i < num_lstm - 1
        layer = LSTM(base_lstm * (2 ** i), return_sequences=not_last_layer, dropout=0.2)(layer)
    if not num_lstm:
        layer = Flatten()(layer)
    for i in range(num_dense):
        layer = Dense(base_dense * (2 ** i), activation='relu')(layer)
        layer = BatchNormalization()(layer)
    output = Dense(1, activation='linear')(layer)  # Output layer
    output = Reshape((1, 1))(output)
    return Model(input, output)

# Define the CNN Discriminator
def build_critic(num_conv, base_conv, num_dense, base_dense, time_step, num_features):
    input = Input(shape=(1, 1))
    layer = input
    for i in range(num_conv - 1, -1, -1):
        layer = Conv1D(base_conv * (2 ** i), kernel_size=3, padding='same', activation='leaky_relu')(layer)
        if i > 0:
            layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    for i in range(num_dense - 1, -1, -1):
        layer = Dense(base_dense * (2 ** i), activation='leaky_relu')(layer)
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

def get_model_paths():
    return 'checkpoints/generator', 'checkpoints/critic'

# Gradient Penalty calculation
def compute_gradient_penalty(real_sequences, fake_sequences, critic):
    # Random weight interpolation between real and fake
    alpha = tf.random.uniform(shape=[real_sequences.shape[0], 1, 1], minval=0., maxval=1.)
    interpolated_sequences = alpha * real_sequences + (1 - alpha) * fake_sequences
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_sequences)
        interpolated_logits = critic(interpolated_sequences)
    
    # Compute gradients w.r.t. the interpolated data
    gradients = tape.gradient(interpolated_logits, interpolated_sequences)
    
    # Compute the norm of the gradients
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    
    # Gradient penalty: ( ||gradients||_2 - 1 )^2
    gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))
    
    return gradient_penalty

def reset_lr(optimizer, lr):
    optimizer.learning_rate.assign(lr)

def adjust_lr(optimizer, decay_factor):
    current_lr = optimizer.learning_rate.numpy()
    new_lr = current_lr * decay_factor
    if new_lr > 5e-1:
        return
    optimizer.learning_rate.assign(new_lr)

    formatted_current_lr = "{:.0e}".format(current_lr).replace("-0", "-")
    formatted_new_lr = "{:.0e}".format(new_lr).replace("-0", "-")
    print(f"Updated learning rate from {formatted_current_lr} to {formatted_new_lr}")
        

# Train the GAN
def train_gan(epochs, batch_size, X, y, num_samples, n_critic, clip_value, gen_lr, critic_lr, num_lstm, gen_dense, base_lstm, gen_base, num_conv, critic_dense, base_conv, critic_base, time_step, num_features, patience, mape_patience, mape_epoch_interval, mape_patience_threshold, mape_plot_threshold, low_mape_epoch_interval, lambda_gp, lambda_mse, restore_checkpoint, decay_factor_critic, decay_factor_gen):
    generator = build_generator(num_lstm, base_lstm, gen_dense, gen_base, time_step, num_features)
    gen_optimizer = Adam(learning_rate=gen_lr)
    gen_checkpoint = tf.train.Checkpoint(model=generator, optimizer=gen_optimizer)
    
    critic = build_critic(num_conv, base_conv, critic_dense, critic_base, time_step, num_features)
    critic_optimizer = Adam(learning_rate=critic_lr)
    critic_checkpoint = tf.train.Checkpoint(model=critic, optimizer=critic_optimizer)

    gen_path, critic_path = get_model_paths()
    gen_checkpoint_manager = tf.train.CheckpointManager(gen_checkpoint, gen_path, max_to_keep=3)
    critic_checkpoint_manager = tf.train.CheckpointManager(critic_checkpoint, critic_path, max_to_keep=3)

    if restore_checkpoint:
        if gen_checkpoint_manager.latest_checkpoint:
            gen_checkpoint.restore(gen_checkpoint_manager.latest_checkpoint)
        if critic_checkpoint_manager.latest_checkpoint:
            critic_checkpoint.restore(critic_checkpoint_manager.latest_checkpoint)
    
    gan_model = build_gan(generator, critic, time_step, num_features)

    critic_losses = []
    generator_losses = []

    # Initialize variables for early stopping and model checkpointing
    best_g_loss = float('inf')
    patience_counter = 0

    best_mape = float('inf')
    last_mape = float('-inf')
    mape_patience_counter = 0
    plot_epoch = False
    best_epoch = 0
    regression_loss = 0
    early_stop_triggered = False
    mape_patience_hitted = False
    generator_weight = 1
    critic_weight = 1
    n_critic_origin = n_critic

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, num_samples, batch_size)
        
            # Real data
            train_data = X[idx].reshape(-1, time_step, num_features)
            real_data = y[idx].reshape(-1, 1, 1)

            # Generate synthetic data
            # noise = tf.random.normal(shape=(batch_size, time_step, num_features))
            fake_data = generator(train_data)

            with tf.GradientTape() as tape:
                real_loss = tf.reduce_mean(critic(real_data))
                fake_loss = tf.reduce_mean(critic(fake_data))
                c_loss = fake_loss - real_loss

                # Compute the gradient penalty
                if lambda_gp:
                    gradient_penalty = compute_gradient_penalty(real_data, fake_data, critic)
                    c_loss = c_loss + lambda_gp * gradient_penalty
                
                weighted_c_loss = c_loss * critic_weight

            grads = tape.gradient(weighted_c_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            # Clip weights if no gradient penalty
            if not lambda_gp:
                for weight in critic.trainable_variables:
                    weight.assign(tf.clip_by_value(weight, -clip_value, clip_value))

        # Train Generator
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(gan_model(train_data)) + regression_loss
            weighted_g_loss = g_loss * generator_weight
            
        grads = tape.gradient(weighted_g_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        critic_losses.append(weighted_c_loss.numpy())
        generator_losses.append(weighted_g_loss.numpy())

        if epoch % mape_epoch_interval == 0:
            features_test_data = get_features_test_data(encoded_features_test, X[-1])
            new_data = generator.predict(features_test_data, verbose=0).flatten()
            mape = evaluate_model(target_test, new_data)
            regression_loss = lambda_mse * tf.reduce_mean(tf.square(tf.cast(fake_data, tf.float32) - tf.cast(real_data, tf.float32)))

            if mape < mape_patience_threshold:
                mape_patience_counter = 0
            else:
                plot_epoch = False
            
            if not mape_plot_threshold or mape < mape_plot_threshold:
                plot_epoch = True
                mape_epoch_interval = low_mape_epoch_interval

            if mape < best_mape:
                # print(f"MAPE% improved: {best_mape - mape} and saved")
                gen_checkpoint_manager.save() # Save the model
                critic_checkpoint_manager.save() # Save the model
                best_mape = mape
                last_mape = mape
                n_critic = n_critic_origin
                best_epoch = epoch
                mape_patience_counter = 0
                if decay_factor_critic:
                    reset_lr(critic_optimizer, critic_lr)
                if decay_factor_gen:
                    reset_lr(gen_optimizer, gen_lr)
            else:
                mape_patience_counter += 1
                critic_weight = 1 + (mape / 50)  # The critic gets stronger with higher MAPE
                generator_weight = 1 / critic_weight  # The generator gets weaker with higher MAPE
                if mape < last_mape:
                    mape_patience_counter = 0
                if mape > last_mape and n_critic < 10:
                    n_critic = n_critic + 1
                    print(f'n_critic increased to {n_critic}')
                elif mape < last_mape and n_critic > n_critic_origin:
                    n_critic = n_critic - 1
                    print(f'n_critic decreased to {n_critic}')

                if decay_factor_critic and mape > last_mape:
                    last_mape = mape
                    adjust_lr(critic_optimizer, decay_factor_critic)
                elif decay_factor_critic > critic_lr and mape < last_mape:
                    adjust_lr(critic_optimizer, 0.8)

                if decay_factor_gen and mape > last_mape:
                    last_mape = mape
                    adjust_lr(gen_optimizer, decay_factor_gen)
                elif decay_factor_gen > gen_lr and mape < last_mape:
                    adjust_lr(gen_optimizer, 0.8)
            
            if plot_epoch:
                predict_origin, test_origin = revert_to_actual_price(new_data, target_test)
                plot_result(predict_origin, test_origin, epoch)
                
            # Early stopping: MAPE
            if mape_patience_counter >= mape_patience:
                mape_patience_hitted = True
                print(f'best_mape {best_mape} at epoch {best_epoch}')
                print("MAPE patience hit.")
                # Exit the training loop
                return (gan_model, generator, critic), (critic_losses, generator_losses), (best_g_loss, best_mape, best_epoch), (early_stop_triggered, mape_patience_hitted)

            # print(f'Epoch {epoch}, Discriminator Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}')

            # Check for improvement
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                patience_counter = 0
                print("Model improved.")
            else:
                patience_counter += 1
            
            # Early stopping: Generator loss
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

def revert_to_actual_price(new_data, target_test):
    # Inverse transform to get the actual predicted price
    predict_origin = scaler_y.inverse_transform(new_data.reshape(-1, 1)).flatten()
    test_origin = scaler_y.inverse_transform(target_test.reshape(-1, 1)).flatten()
    return predict_origin, test_origin

# Evaluate the model using multiple metrics
def evaluate_model(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    # print("Evaluation Metrics:")
    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Mean Absolute Error (MAE): {mae}")
    # print(f"R-squared (R²): {r2}")
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

reduce_index = 1
num_samples, time_step, batch_size, batch_sizes = get_hyperparams(time_step=time_step, features_train=features_train, reduce_index=reduce_index)
# batch_size = 8

train_data, train_target = prepare_data(num_samples=num_samples, time_step=time_step, features_train=features_train, target_train=target_train)
X = train_data
y = train_target

# This block will only execute when this file is run directly
if __name__ == "__main__":

    patience = 15
    mape_patience = 5
    mape_epoch_interval = 10 # MAPE will be check on this inverval of epoch
    mape_patience_threshold = 20 # While mape get lower than this value, mape break will be disabled
    mape_plot_threshold = 15 # A flag to show preview plot will be set when mape passed down this value, then the preview will be shown on every next mape_epoch_interval. Setting this value to 0 will show preview on every mape_epoch_interval regardless of mape value.
    low_mape_epoch_interval = 10 # Reduce mape_epoch_interval to this value to check MAPE more often when the result get closer to actual
    num_epoch = 4000

    # Learning rates
    gen_lr = 1e-5
    critic_lr = 1e-4

    n_critic = 4 # Number of training steps for the critic per generator step
    clip_value = 0.01
    lambda_gp = 9 # Gradient penalty weight
    lambda_mse = 0 # 0.1 to 10
    
    # Generator
    num_lstm = 0
    base_lstm = 32

    gen_dense = 1
    gen_base = 64

    # Critic
    num_conv = 3
    base_conv = 64

    critic_dense = 1
    critic_base = 32

    restore_checkpoint = False
    decay_factor_gen = 1.5
    decay_factor_critic = 0
    
    # plot_train(features_train, target_train)

    def automate_train():
        global restore_checkpoint
        models, losses, bests, breaks = train_gan(num_epoch, batch_size, X, y, num_samples, n_critic, clip_value, gen_lr, critic_lr, num_lstm, gen_dense, base_lstm, gen_base, num_conv, critic_dense, base_conv, critic_base, time_step, num_features, patience, mape_patience, mape_epoch_interval, mape_patience_threshold, mape_plot_threshold, low_mape_epoch_interval, lambda_gp, lambda_mse, restore_checkpoint, decay_factor_critic=decay_factor_critic, decay_factor_gen=decay_factor_gen)
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

    predict_origin, test_origin = revert_to_actual_price(new_data, target_test)

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
        plot_result(predict_origin, test_origin)

    visualize_result()