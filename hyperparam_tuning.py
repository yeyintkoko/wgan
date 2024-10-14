import optuna
from gan import train_gan, num_epoch, batch_sizes, X, y, num_samples, time_step, num_features

# Define an Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    gan_lr = trial.suggest_float('gan_lr', 1e-5, 1e-1)
    critic_lr = trial.suggest_float('critic_lr', 1e-5, 1e-1)
    n_critic = trial.suggest_int('n_critic', 1, 25)
    clip_value = trial.suggest_float('clip_value', 0.001, 1)
    num_lstm = trial.suggest_int('num_lstm', 1, 10)
    num_lstm_dense = trial.suggest_int('num_lstm_dense', 1, 10)
    num_conv = trial.suggest_int('num_conv', 1, 10)
    num_conv_dense = trial.suggest_int('num_conv_dense', 1, 10)
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    
    # Train GAN and get the best generator loss
    _, _, best_g_loss = train_gan(epochs=num_epoch, batch_size=batch_size, X=X, y=y, num_samples=num_samples, n_critic=n_critic, clip_value=clip_value, gan_lr=gan_lr, critic_lr=critic_lr, num_lstm=num_lstm, num_lstm_dense=num_lstm_dense, num_conv=num_conv, num_conv_dense=num_conv_dense, time_step=time_step, num_features=num_features)
    return best_g_loss

# Create a study and optimize the objective
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)