import numpy as np
import pandas as pd

# Step 2: Read the CSV file into a DataFrame
df = pd.read_csv('data/output.csv')

# Step 3: Convert the DataFrame to a NumPy array
numpy_array = df.to_numpy(dtype=float)

# Set print options for NumPy
np.set_printoptions(suppress=True, precision=6)

# Print the original DataFrame and the NumPy array
print("Original DataFrame:")
print(df)

print("\nConverted NumPy Array:")
print(numpy_array)