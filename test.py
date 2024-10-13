import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

batch_sizes = get_divisors(1695)
print('batch_sizes', batch_sizes)

divider = len(batch_sizes) // 2
print('divider', divider)

batch_size = batch_sizes[divider]
print('batch_size', batch_size)
