from pandas_datareader import data as pdr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load df
df = pd.read_csv('data/output.csv', header=0)
df = df[['price', 'ma7', '26ema', '12ema', 'upper_band', 'lower_band', 'ema', 'momentum']]
df = df[::-1].reset_index(drop=True)

# Define number of training days
num_training_days = int(df.shape[0] * 0.9)

print(df.head())

def plot_data():
    sns.set_style('darkgrid')
    plt.figure(figsize=(14,5),dpi=120)
    sns.lineplot(df, x=df.index, y=df[['price']].squeeze())
    plt.show()

def normalize_cols(df,cols):
    """Scale the values of each feature
    according to the columns max value"""
    data = df.loc[:,cols]
    for col in cols:
        scaler = lambda x: x / data[col].max()
        data[col] = data[col].apply(scaler)
    print(data[cols].head())
    return data[cols].values

features = df.columns.values[:] # columns to train model on
X = normalize_cols(df,features)

"""
### Turn each signal into a labeled dataset
"""
window_size = 30   # num. days per training sample
batch_size = 128   # num. of samples per epoch
buffer_size = 1000 # num of samples in memory for random selection
split_time = num_training_days  # where to split the data for training/validation


def window_dataset(series, window_size, batch_size, shuffle_buffer):
    """Funtion to turn time series data into set of sequences 
    where the last value is the intended output of our model"""
    ser = tf.expand_dims(series, axis=-1)
    data = tf.data.Dataset.from_tensor_slices(series)
    data = data.window(window_size + 1, shift=1, drop_remainder=True)
    data = data.flat_map(lambda w: w.batch(window_size + 1))
    data = data.shuffle(shuffle_buffer)
    data = data.map(lambda w: (w[:-1], w[1:]))
    return data.batch(batch_size).prefetch(1)


x_train = X[:split_time,:]
x_test = X[split_time:,:]

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_test.shape}")

train_set = window_dataset(x_train,window_size,batch_size,buffer_size)

keras.backend.clear_session()

"""
### Choose and connect the model components   
"""
# 1D convolution layers
conv1 = layers.Conv1D(
    filters=60,kernel_size=15,strides=1,
    padding="causal",activation="relu")

conv2 = layers.Conv1D(
    filters=60,kernel_size=5,strides=1,
    padding="causal",activation="tanh")

# Bidirectional LSTM layers
lstm1 = layers.Bidirectional(layers.LSTM(30,return_sequences=True))
lstm2 = layers.Bidirectional(layers.LSTM(20,return_sequences=True))

# Model construction
inputs = layers.Input(shape=(None,len(features)))
x = conv1(inputs)
x = lstm1(x)
x = lstm2(x)
x = conv2(x)
x = layers.Dense(60,activation='relu')(x)
x = layers.Dropout(.1)(x)
x = layers.Dense(1,activation='tanh')(x)
outputs = layers.Lambda(lambda x: 25*abs(x))(x)

# SGD optimizer and Huber loss
optimizer = keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
loss = keras.losses.Huber()

model = keras.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer,loss,
              metrics=["mae"])
model.summary()

"""
### Train model
"""
epochs = 1500

history = model.fit(train_set, epochs=epochs, verbose=0)
print(f"Model trained for {epochs} epochs")


"""
### Inspect training results
"""
def model_forecast(model, X, window_size):
    """Takes in numpy array, creates a windowed tensor 
    and predicts the following value on each window"""
    data = tf.data.Dataset.from_tensor_slices(X)
    data = data.window(window_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda w: w.batch(window_size))
    data = data.batch(32).prefetch(1)
    forecast = model.predict(data)
    return forecast

train_window = [i for i in range(split_time-window_size)]
train_window_df = pd.DataFrame(train_window, columns=['Date'])

forecast = model_forecast(model,x_train,window_size)

# Plot the forecast vs actual data
def plot_prediction():
    plt.figure(figsize=(14, 5), dpi=120)
    sns.lineplot(data=train_window_df, x='Date', y=forecast[:-1, -1, 0], label='Forecast')
    sns.lineplot(data=train_window_df, x='Date', y=X[:split_time - window_size, 0], label='Actual')
    plt.legend()
    plt.show()
plot_prediction()

"""
### Make predictions on unseen data
"""
val_window = [i for i in range(split_time,len(df)-window_size)]
val_window_df = pd.DataFrame(val_window, columns=['Date'])

forecast = model_forecast(model,x_test,window_size)

def plot_forecast():
    plt.figure(figsize=(8,5),dpi=120)
    sns.lineplot(data=val_window_df, x='Date', y=forecast[:-1, -1, 0], label='Forecast')
    sns.lineplot(data=val_window_df, x='Date', y=X[split_time:-window_size, 0], label='Actual')
    plt.legend()
    plt.show()
plot_forecast()