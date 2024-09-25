# ---------- faltten test --
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
inputs = Input(shape=(3,2,4))

# Define a model consisting only of the Flatten operation
prediction = Flatten()(inputs)
model = Model(inputs=inputs, outputs=prediction)

X = np.arange(0,24).reshape(1,3,2,4)
print(X)
model.predict(X)

# --------------- 2 inputs  1 outputs test

#Dummy data (USE YOUR OWN DATA HERE AS NUMPY ARRAYS
import numpy as np
X1 = np.random.random((1000, 3, 4))
X2 = np.random.random((1000, 2, 2))
y = np.random.randint(0, 2, (1000,))

#Scaling individual features by respective min max for 3D tensors
from sklearn.preprocessing import MinMaxScaler

#Have separate scaler objects for each input data
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

#Ensure that in reshape to 2D matrix, you keep the number of features separate
#as min-max scaler works on each feature's respective min-max values
#Then, reshape it back to the 3D dataset

X1_scaled = scaler1.fit_transform(X1.reshape(-1,X1.shape[-1])).reshape(X1.shape)
X2_scaled = scaler1.fit_transform(X2.reshape(-1,X2.shape[-1])).reshape(X2.shape)

print(X1_scaled.shape, X2_scaled.shape)

from sklearn.model_selection import train_test_split

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1_scaled, X2_scaled, y, test_size=0.2)

print([i.shape for i in (X1_train, X1_test, X2_train, X2_test, y_train, y_test)])

from tensorflow.keras import layers, Model, utils

inp1 = layers.Input((3,4))
inp2 = layers.Input((2,2))
x1 = layers.Flatten()(inp1)
x2 = layers.Flatten()(inp2)
x = layers.concatenate([x1, x2])
x = layers.Dense(32)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = Model([inp1, inp2], out)

# utils.plot_model(model, show_layer_names=False, show_shapes=True)

# Train
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit([X1_train, X2_train], y_train, epochs=4)