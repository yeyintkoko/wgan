import time
import numpy as np
import pandas as pd  # Import pandas
from pandas.plotting import autocorrelation_plot
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.initializers import HeNormal

import math
from collections import deque

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# just import bert
import bert

import warnings
warnings.filterwarnings("ignore")

def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')

dataset_ex_df = pd.read_csv('data/panel_data_close.csv', header=0, parse_dates=[0], date_parser=parser)

dataset_ex_df[['Date', 'price']].head(3)

print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

num_training_days = int(dataset_ex_df.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, dataset_ex_df.shape[0]-num_training_days))

def plot_stock_price(dataset):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(dataset['Date'], dataset['price'], label='Apple stock')
    plt.vlines(datetime.date(2016,4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.title('Figure 1: Apple stock price')
    plt.legend()
    plt.show()

def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['price'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['price'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['price'].rolling(window=20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['price'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['price']-1

    # Create Log Momentum
    dataset['log_momentum'] = np.log(dataset['price'] / dataset['price'].shift(1))
    
    return dataset

dataset_TI_df = get_technical_indicators(dataset_ex_df[['price']])
dataset_TI_df.head()

# Save the DataFrame to a CSV file
dataset_TI_df.to_csv('data/output.csv', index=False)

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['price'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Apple - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')

    plt.title('Figure 2: Apple (close) stock prices & Technical Indicators')
    plt.legend()
    plt.show()

def get_fft_dataframe(dataset):
    close_fft = np.fft.fft(np.asarray(dataset['price'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    return fft_df

def plot_fourier_transforms(dataset):
    fft_df = get_fft_dataframe(dataset)
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(dataset['price'],  label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Apple (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()

def plot_wavelets(dataset):
    fft_df = get_fft_dataframe(dataset)
    items = deque(np.asarray(fft_df['absolute'].tolist()))
    items.rotate(int(np.floor(len(fft_df)/2)))
    plt.figure(figsize=(10, 7), dpi=80)
    plt.stem(items)
    plt.title('Figure 4: Components of Fourier transforms')
    plt.show()

data_FT = dataset_ex_df[['Date', 'price']]

def print_arima_model(dataset):
    series = dataset['price']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    print(model_fit.summary())

def plot_auto_correlation(dataset):
    series = dataset['price']
    autocorrelation_plot(series)
    plt.show()

def predict_value(dataset):
    series = dataset['price']
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    plot_predicted_values(test, predictions)
    
def plot_predicted_values(test, predictions):
    # Plot the predicted (from ARIMA) and real prices
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(test, label='Real')
    plt.plot(predictions, color='red', label='Predicted')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 5: ARIMA model on Apple stock')
    plt.legend()
    plt.show()

def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['price']
    X = data.iloc[:, 1:]
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    return (X_train, y_train), (X_test, y_test)

def get_regression_test_result(dataset):
    # Get training and test data
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset)

    # Get xgboost for regression test
    regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)

    xgbModel = regressor.fit(X_train_FI,y_train_FI, eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], verbose=False)

    eval_result = regressor.evals_result()

    training_rounds = range(len(eval_result['validation_0']['rmse']))

    return training_rounds, eval_result, xgbModel, (X_train_FI, y_train_FI), (X_test_FI, y_test_FI)

def plot_regression_test(dataset):
    training_rounds, eval_result, _, _, _ = get_regression_test_result(dataset)
    plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
    plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.show()

def plot_feature_importances(dataset):
    _, _, xgbModel, _, (X_test_FI, _) = get_regression_test_result(dataset)
    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.show()

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
def relu(x):
    return max(x, 0)
def lrelu(x):
    return max(0.01*x, x)


def plot_gaussian_error():
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

    ranges_ = (-10, 3, .25)

    plt.subplot(1, 2, 1)
    plt.plot([i for i in np.arange(*ranges_)], [relu(i) for i in np.arange(*ranges_)], label='ReLU', marker='.')
    plt.plot([i for i in np.arange(*ranges_)], [gelu(i) for i in np.arange(*ranges_)], label='GELU')
    plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
    plt.title('Figure 7: GELU as an activation function for autoencoders')
    plt.ylabel('f(x) for GELU and ReLU')
    plt.xlabel('x')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([i for i in np.arange(*ranges_)], [lrelu(i) for i in np.arange(*ranges_)], label='Leaky ReLU')
    plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
    plt.ylabel('f(x) for Leaky ReLU')
    plt.xlabel('x')
    plt.title('Figure 8: LeakyReLU')
    plt.legend()

    plt.show()


# plot_stock_price(dataset_ex_df)
# plot_technical_indicators(dataset_TI_df, 400)
# plot_fourier_transforms(data_FT)
# plot_wavelets(data_FT)
# print_arima_model(data_FT)
# plot_auto_correlation(data_FT)
# predict_value(data_FT)
# plot_regression_test(dataset_TI_df)
# plot_feature_importances(dataset_TI_df)
# plot_gaussian_error()

#-----------
batch_size = 64
VAE_data = dataset_TI_df
VAE_data = VAE_data.dropna() # drop rows with empty columns (only first 26 rows)

# Update number of days after empty rows has been removed
num_training_days = int(VAE_data.shape[0]*.7)
print('Updated: Number of training days: {}. Number of test days: {}.'.format(num_training_days, VAE_data.shape[0]-num_training_days))

train_steps_per_epoch = num_training_days // batch_size

# Calculate the maximum number of samples that can fit into full batches
max_full_batches = train_steps_per_epoch * batch_size

num_testing_days = VAE_data.shape[0]-max_full_batches

test_steps_per_epoch = (num_testing_days) // batch_size
print('steps per epoch: {}, tests per epoch: {}'.format(train_steps_per_epoch, test_steps_per_epoch))

max_testing_days = test_steps_per_epoch * batch_size

end_position = num_testing_days - max_testing_days

VAE_data = VAE_data.values

# Split the data into training and testing sets
train_data = VAE_data[:max_full_batches, :-1]
train_labels = VAE_data[:max_full_batches, -1]
test_data = VAE_data[max_full_batches:-end_position, :-1]
test_labels = VAE_data[max_full_batches:-end_position, -1]

def create_dataset(data, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset

# Create TensorFlow datasets
train_dataset = create_dataset(train_data, train_labels, batch_size)
test_dataset = create_dataset(test_data, test_labels, batch_size)

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.n_latent = n_latent
        
        # Encoder
        self.encoder = models.Sequential()
        for _ in range(n_layers):
            self.encoder.add(layers.Dense(n_hidden, activation='gelu', kernel_initializer=HeNormal()))
        self.encoder.add(layers.Dense(n_latent * 2, activation='sigmoid', kernel_initializer=HeNormal()))  # Outputs mean and log variance

        # Decoder
        self.decoder = models.Sequential()
        for _ in range(n_layers):
            self.decoder.add(layers.Dense(n_hidden, activation='gelu', kernel_initializer=HeNormal()))
        self.decoder.add(layers.Dense(n_output, activation='sigmoid', kernel_initializer=HeNormal()))

    # Calculate loss in the call method
    def call(self, inputs):
        # Encode
        h = self.encoder(inputs)
        mu, lv = tf.split(h, num_or_size_splits=2, axis=1)

        # Reparameterization trick
        eps = tf.random.normal(shape=(tf.shape(inputs)[0], self.n_latent))
        z = mu + tf.exp(0.5 * lv) * eps

        # Decode
        outputs = self.decoder(z)

        # Ensure outputs are in the correct shape
        assert outputs.shape[1] == inputs.shape[1], f"Output shape {outputs.shape[1]} does not match input shape {inputs.shape[1]}"

        # Calculate loss
        kl_loss = -0.5 * tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1)
        logloss = tf.reduce_sum(inputs * tf.math.log(outputs + 1e-10) + 
                                (1 - inputs) * tf.math.log(1 - outputs + 1e-10), axis=1)
        total_loss = -logloss - kl_loss

        self.add_loss(tf.reduce_mean(total_loss))
        return outputs

# Initialize the VAE model
n_hidden = 400
n_latent = 2
n_layers = 3
n_output = VAE_data.shape[1] - 1

vae = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output)
# Create trainer with lower training rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
vae.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
# vae.fit(train_dataset, steps_per_epoch=len(train_data) // batch_size, epochs=50)

# To evaluate the model
# predictions = vae.predict(test_data)

n_epoch = 150
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []

epoch_loop_count = 0
for epoch in range(n_epoch):
    epoch_loss = 0
    epoch_val_loss = 0
    epoch_loop_count += 1

    n_batch_train = 0
    for batch in train_dataset:
        n_batch_train += 1
        data, labels = batch  # Unpack the data and labels

        with tf.GradientTape() as tape:
            loss = vae(data)  # Forward pass

        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))  # Update weights

        epoch_loss += tf.reduce_mean(loss)

    n_batch_val = 0
    for batch in test_dataset:
        n_batch_val += 1
        data, labels = batch  # Unpack the data and labels
        loss = vae(data)  # Forward pass
        epoch_val_loss += tf.reduce_mean(loss)

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss.numpy())
    validation_loss.append(epoch_val_loss.numpy())

    print('Epoch {}, Training loss {}, Validation loss {}\nTrained count {}, Tested count {}'.format(epoch, epoch_loss.numpy(), epoch_val_loss.numpy(), n_batch_train, n_batch_val))
        

end = time.time()
print('Training completed in {} seconds.'.format(int(end - start)))

vae.summary()

print(vae)

# # Print encoder layers
# print("Encoder layers:")
# for layer in vae.encoder.layers:
#     print(f"Layer Name: {layer.name}")
#     print(f"Layer Type: {layer.__class__.__name__}")
#     print(f"Output Shape: {layer.output}")
#     print(f"Number of Parameters: {layer.count_params()}")
#     print("Weights and Biases:")
#     print(layer.get_weights())  # This will show the weights and biases for the layer
#     print("-" * 40)

# # Print decoder layers
# print("Decoder layers:")
# for layer in vae.decoder.layers:
#     print(f"Layer Name: {layer.name}")
#     print(f"Layer Type: {layer.__class__.__name__}")
#     print(f"Output Shape: {layer.output}")
#     print(f"Number of Parameters: {layer.count_params()}")
#     print("Weights and Biases:")
#     print(layer.get_weights())  # This will show the weights and biases for the layer
#     print("-" * 40)

# dataset_total_df['Date'] = dataset_ex_df['Date']
# vae_added_df = mx.nd.array(dataset_total_df.iloc[:, :-1].values)
# print('The shape of the newly created (from the autoencoder) features is {}.'.format(vae_added_df.shape))

# Plot the results
# plt.figure(figsize=(14, 5))
# plt.plot(test_data, color='blue', label='Actual Stock Price')
# plt.plot(np.arange(len(train_data), len(train_data) + len(predictions)), predictions, color='red', label='Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# ----------------





# # We want the PCA to create the new components to explain 80% of the variance
# pca = PCA(n_components=.8)


def lstm_test():
    # Load the dataset
    data = pd.read_csv('data/panel_data_close.csv')  # Ensure this CSV has a 'price' column

    # Use the 'Close' price for prediction
    prices = data['price'].values
    prices = prices.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)

    # Prepare the training data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    # Create datasets for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60  # Number of previous days to consider
    X_train, y_train = create_dataset(train_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64)

    # Prepare test data
    test_data = scaled_data[train_size - time_step:]
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Inverse scaling

    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(data['price'].values, color='blue', label='Actual Stock Price')
    plt.plot(np.arange(len(train_data) + time_step, len(train_data) + len(predictions) + time_step), predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# lstm_test()


