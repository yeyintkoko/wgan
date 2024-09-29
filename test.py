import numpy as np
import pandas as pd
from autoencoder import scaler_X, num_features

noise = np.random.uniform(10, 300, num_features)
print(noise)

X_train = scaler_X.fit_transform(noise.reshape(-1, 1))

print(X_train)