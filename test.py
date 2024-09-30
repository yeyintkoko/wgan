import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# np.set_printoptions(suppress=True, precision=6)

scaler_y = StandardScaler()

change_rate = scaler_y.fit_transform(np.array('6.8336185e-08').reshape(-1,1)).flatten()
print(change_rate)

change_rate = scaler_y.transform(np.array('-1.3667237e-07').reshape(-1,1)).flatten()
print(change_rate)

change_rate = scaler_y.transform(np.array('1.3667238e-07').reshape(-1,1)).flatten()
print(change_rate)

change_rate = scaler_y.transform(np.array('-0.0').reshape(-1,1)).flatten()
print(change_rate)