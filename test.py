import numpy as np
import pandas as pd
from autoencoder import autocorrelation

origin = [
    [ 89.640366,  66.785227,  65.009476, 133.688806,  47.827895,  22.64049 ],
    [89.74601,  42.452486, 43.012926,  0.56044,  43.201378, 42.805007],
    [89.723798, 42.441979, 43.00228,   0.560301, 43.190686, 42.794413],
    [89.728452, 42.434426, 42.994628,  0.560201, 43.183,    42.786797],
    [89.730918, 42.428899, 42.989028,  0.560128, 43.177375, 42.781224]
]

# Normalizing the data before calculating autocorrelation
def normalize(data):
    return (data - np.mean(data)) / np.std(data)

# Modify your function to normalize data
def generate_new_feature(last_data):
    last_data = np.array(last_data)
    normalized_data = normalize(last_data)
    autocorr_values = autocorrelation(pd.DataFrame(normalized_data), lag=1)
    weights = np.array(autocorr_values).flatten()
    absolute_weights = np.abs(weights)
    weighted_features = last_data * absolute_weights
    return weighted_features

new_features = origin
for i in range(1,10):
    new_features = generate_new_feature(new_features)
    print('new_features', new_features)
