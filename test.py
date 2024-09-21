import numpy as np

# Assume VAE_data is your input data
# If VAE_data is a pandas DataFrame, convert it to a numpy array
VAE_data = np.random.rand(100, 1).astype(np.float32)  # Example data with 1000 samples
batch_size = 5
n_output = VAE_data.shape[1] - 1  # Adjust if necessary

print('----- data ---- {}'.format(VAE_data))

# Create a batch of data
# train = VAE_data[:batch_size,1:]  # Get the first 'batch_size' samples
# real = VAE_data[batch_size:,:1]
# print('----- train ---- {}'.format(train))
# print('----- real ---- {}'.format(real))

arr = [1,2,3,4,5,6,7]
i = 0
test_step = 2
print('----- test ---- {}'.format(arr[i + 2]))
print('----- test ---- {}'.format(arr[0]))
print('----- test ---- {}'.format(arr[1]))
print('----- test ---- {}'.format(arr[6]))