import time
import numpy as np
import pandas as pd  # Import pandas

# from mxnet import nd, autograd, gluon
# from mxnet.gluon import nn, rnn
# import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
# %matplotlib inline  # Comment out or remove if not using a Jupyter notebook
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# context = mx.cpu()
# model_ctx = mx.cpu()
# mx.random.seed(1719)

def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')

dataset_ex_df = pd.read_csv('data/panel_data_close.csv', header=0, parse_dates=[0], date_parser=parser)

dataset_ex_df[['Date', 'GS']].head(3)

print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_ex_df['Date'], dataset_ex_df['GS'], label='Goldman Sachs stock')
plt.axvline(pd.to_datetime('2016-04-20'), color='gray', linestyle='--', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 2: Goldman Sachs stock price')
plt.legend()
plt.show()
