import numpy as np

# Assume VAE_data is your input data
# If VAE_data is a pandas DataFrame, convert it to a numpy array
VAE_data = np.random.rand(10, 1).astype(np.float32)  # Example data with 1000 samples
batch_size = 5
n_output = VAE_data.shape[1] - 1  # Adjust if necessary

# print(VAE_data)

# Create a batch of data
data = VAE_data[:batch_size]  # Get the first 'batch_size' samples
dbtb = VAE_data[batch_size:]
print(data)
print('----------')
print(dbtb)
