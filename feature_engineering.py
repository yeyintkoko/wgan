import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

dataset = pd.read_csv("data/output.csv", header=0).dropna()
dataset = dataset[['price', 'ma7', '26ema', '12ema', 'MACD', 'ema', 'momentum']]
dataset = dataset[::-1].reset_index(drop=True)

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
plt.plot(dataset['26ema'],label='MA 26', color='r',linestyle='--')
plt.plot(dataset['12ema'],label='MA 12', color='b')
plt.plot(dataset['ema'],label='EMA', color='orange',linestyle='--')
plt.plot(dataset['momentum'],label='Momentum', color='purple',linestyle='--')
plt.title('Technical indicators for Apple.')
plt.ylabel('USD')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(dataset['price'], label='Price', color='green')
plt.title('Price Returns Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Correlation Analysis
correlation = dataset['MACD'].corr(dataset['price'])
print(f'Correlation between sentiment and returns: {correlation}')

# 6. Statistical Testing
correlation_coefficient, p_value = pearsonr(dataset['MACD'].dropna(), dataset['price'].dropna())
print(f'Pearson correlation coefficient: {correlation_coefficient}, p-value: {p_value}')