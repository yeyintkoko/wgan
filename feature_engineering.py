import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

dataset = pd.read_csv("data/output.csv", header=0).dropna()
dataset = dataset[['price', 'ma7', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'momentum']]
dataset = dataset[::-1].reset_index(drop=True)

last_days = 600 #Plot for last 600 days
def plot_dataset():
    plt.figure(figsize=(14, 7))
    shape_0 = dataset.shape[0]
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['26ema'],label='MA 26', color='r',linestyle='--')
    plt.plot(dataset['12ema'],label='MA 12', color='b')
    plt.plot(dataset['ema'],label='EMA', color='orange',linestyle='--')
    plt.plot(dataset['momentum'],label='Momentum', color='purple',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Apple last 600 days.')
    plt.ylabel('USD')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dataset['price'], label='Price', color='green')
    plt.title('Price Returns Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# plot_dataset()

# Assuming 'dataset' is your DataFrame
correlation_results = {}

# Calculate correlation for all features against 'price'
for feature in dataset.columns:
    if feature != 'price':  # Skip the price column itself
        correlation = dataset[feature].corr(dataset['price'])
        correlation_coefficient, p_value = pearsonr(dataset[feature].dropna(), dataset['price'].dropna())
        correlation_results[feature] = {
            'correlation': correlation,
            'pearson_coefficient': correlation_coefficient,
            'p_value': p_value
        }

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(correlation_results).T
results_df = results_df.reset_index().rename(columns={'index': 'feature'})

# Print the correlation results
print(results_df)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.bar(results_df['feature'], results_df['correlation'], color='skyblue')
plt.axhline(0, color='red', linestyle='--')  # Reference line at y=0
plt.xticks(rotation=45)
plt.title('Correlation of Features with Price')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()