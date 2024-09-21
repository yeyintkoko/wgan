import numpy as np

# Assume VAE_data is your input data
# If VAE_data is a pandas DataFrame, convert it to a numpy array

VAE_data = np.random.rand(10, 3).astype(np.float32)  # Example data with 1000 samples
batch_size = 5
n_output = VAE_data.shape[1] - 1  # Adjust if necessary

print('----- data ---- {}'.format(VAE_data))

print('----- VAE_data.shape ---- {}'.format(VAE_data.shape))

VAE_data = VAE_data.reshape(VAE_data.shape[0], VAE_data.shape[1], 1)

print('----- after VAE_data.shape ---- {}'.format(VAE_data.shape))

print('----- data after reshape---- {}'.format(VAE_data))