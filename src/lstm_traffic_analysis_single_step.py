# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

# Importing the training setx
dataset_train = pd.read_csv('traffic_data.csv')
training_set = dataset_train.iloc[:, 5:6].values

from sklearn.model_selection import train_test_split
X_train_split, X_test_split = train_test_split(training_set, shuffle=False, test_size = 0.2, random_state = 0)
X_train_split = pd.DataFrame(X_train_split)
X_test_split = pd.DataFrame(X_test_split)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(X_train_split)

timelag = 24
X_train = []
y_train = []
for i in range(timelag, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - timelag : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
'''
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 20), input_shape = (X_train.shape[1], 1))
regressor.add(Dropout(0.2))
'''
# Adding the output layer
regressor.add(Dense(units = 1,activation='relu'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['mae'])

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 256)

dataset_test = pd.read_csv('traffic_data_test.csv')
real_volume = dataset_test.iloc[:, 5:6].values


dataset_total = pd.concat((X_train_split , X_test_split), axis = 0)
inputs = dataset_total[len(dataset_total) - len(X_test_split) - timelag:].values
inputs = inputs.reshape(-1,1)

# don't need to use fit as it's fitted already
inputs = sc.transform(inputs)
X_test = []
for i in range(timelag, len(inputs)):
    X_test.append(inputs[i - timelag : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_volume = regressor.predict(X_test)
predicted_volume = sc.inverse_transform(predicted_volume)

# Visualising the results
plt.plot(X_test_split, color = 'red', label = 'Real volume in Lane-2')
plt.plot(predicted_volume, color = 'blue', label = 'Predicted volume in Lane-2')
plt.title('Volume in Lane-2 prediction')
plt.xlabel('Time')
plt.ylabel('Volume')
leg = plt.legend()
plt.show()