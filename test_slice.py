import numpy as np
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 6, 0, 1, 8, 2, 9, 4],
    'B': [4, 5, 6, 4, 2, 4, 2, 7, 0, 1],
    'C': [7, 8, 9, 3, 5, 5, 6, 2, 8, 0]
})

# print(df)
# print(df.shape)

num_feature = df.shape[1]
time_step = 3
num_samples = len(df) // time_step
df = df[:num_samples * time_step]
df = np.array(df)
df = df.reshape(num_samples, time_step, num_feature)
# print('---- df after resahped ----')
# print(df)
# print(df.shape)

X = df[:, :-1, :]
# print(X)

y = df[:, -1, 0]
# print(y)