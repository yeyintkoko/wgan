import numpy as np
import pandas as pd
import datetime
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assume VAE_data is your input data
# If VAE_data is a pandas DataFrame, convert it to a numpy array

# VAE_data = np.random.rand(10, 3).astype(np.float32)  # Example data with 1000 samples
# batch_size = 5
# n_output = VAE_data.shape[1] - 1  # Adjust if necessary

# print('----- data ---- {}'.format(VAE_data))

# print('----- VAE_data.shape ---- {}'.format(VAE_data.shape))

# VAE_data = VAE_data.reshape(VAE_data.shape[0], VAE_data.shape[1], 1)

# print('----- after VAE_data.shape ---- {}'.format(VAE_data.shape))

# print('----- data after reshape---- {}'.format(VAE_data))
 
# def parser(x):
#     return datetime.datetime.strptime(x, '%m/%d/%Y')

# # read the data from the csv
# dataset = pd.read_csv("data/panel_data_close.csv",
#                    header=0, index_col=0, parse_dates=[0], date_format=parser)

# pd.DataFrame.corr(method = 'pearson')

# extracting only the temperature values
# values = pd.DataFrame(data.values)
 
# using shift function to shift the values.
# dataframe = pd.concat([values.shift(3), values.shift(2),
#                        values.shift(1), values], axis=1)
# # naming the columns
# dataframe.columns = ['t', 't+1', 't+2', 't+3']
 
# # using corr() function to compute the correlation
# result = dataframe.corr()
 
# print(result)
 
# # plot the auto correlation
#  plot_acf(data)

# time_lag=4
# dataframe=data[['price']]
# x_train=dataframe[0:-time_lag]
# y_train=dataframe[time_lag:]

# result = autocorrelation_plot(dataframe)
# print(result)
# plt.show()



# Prepare input X (all features) and output y (only price)
# X = data[:, :-1, :]  # All but the last time step as input
# y = data[:, 1:, 0]   # Only price for the next time step

