import optuna
from gan import train_gan, X, y, num_samples, time_step, num_features

# Define an Optuna objective function
def objective(trial):
    try:
        # Suggest hyperparameters
        lrs = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
        gan_lr = trial.suggest_categorical('gan_lr', lrs)
        critic_lr = trial.suggest_categorical('critic_lr', lrs)
        n_critic = trial.suggest_int('n_critic', 1, 20)
        clip_value = 0.01
        num_lstm = trial.suggest_int('num_lstm', 1, 4)
        num_lstm_dense = trial.suggest_int('num_lstm_dense', 1, 6)
        num_conv = trial.suggest_int('num_conv', 1, 8)
        num_conv_dense = trial.suggest_int('num_conv_dense', 1, 6)
        batch_size = 18
        num_epoch = 150
        num_lstm_hidden = trial.suggest_categorical('num_lstm_hidden', [32, 50, 64, 80, 100, 128, 150, 200, 256])
        num_lstm_base = trial.suggest_categorical('num_lstm_base', [16, 24, 32, 40, 48])
        num_conv_base = trial.suggest_categorical('num_conv_base', [16, 24, 32, 40, 48])

        # Train GAN and get the best generator loss
        _, _, best_g_loss = train_gan(epochs=num_epoch, batch_size=batch_size, X=X, y=y, num_samples=num_samples, n_critic=n_critic, clip_value=clip_value, gan_lr=gan_lr, critic_lr=critic_lr, num_lstm=num_lstm, num_lstm_dense=num_lstm_dense, num_lstm_hidden=num_lstm_hidden, num_lstm_base=num_lstm_base, num_conv=num_conv, num_conv_dense=num_conv_dense, num_conv_base=num_conv_base, time_step=time_step, num_features=num_features)
        return best_g_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')  # Return a high value for failed trials
    
# Create a study and optimize the objective
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)