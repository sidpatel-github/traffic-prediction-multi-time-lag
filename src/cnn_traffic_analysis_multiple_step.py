# Importing the Keras libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Conv1D, Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense


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

timelag = 6
n_future = 3
X_train = []
y_train = []
for i in range(timelag, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - timelag : i, 0])
    y_train.append(training_set_scaled[i  : i + n_future, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialising the CNN
classifier = Sequential()

classifier.add(Conv1D(64, 3, padding="same",activation = 'relu', input_shape = (X_train.shape[1], 1)))
#classifier.add(MaxPooling1D(2, padding="same"))

# Adding a second convolutional layer This is to increase the efficiency
classifier.add(Conv1D(64, 3,padding="same", activation = 'relu'))
classifier.add(MaxPooling1D(2, padding="same"))

classifier.add(Conv1D(64, 3,padding="same", activation = 'relu'))
classifier.add(MaxPooling1D(2, padding="same"))

classifier.add(Conv1D(64, 3,padding="same", activation = 'relu'))
classifier.add(MaxPooling1D(2, padding="same"))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(128, activation='relu'))

classifier.add(Dense(units = 3))

classifier.summary()
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['mae'])

classifier.fit(X_train, y_train, epochs = 100, batch_size = 256)

# Part 2

real_volume = []
for i in range(0, len(X_test_split) - n_future + 1):
    real_volume.append(X_test_split.values[i  : i + n_future, 0])
    
real_volume = np.array(real_volume)

dataset_total = pd.concat((X_train_split , X_test_split), axis = 0)
inputs = dataset_total[len(dataset_total) - len(X_test_split) - timelag:].values
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []
for i in range(timelag, len(inputs)):
    X_test.append(inputs[i - timelag : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_volume = classifier.predict(X_test)
predicted_volume = sc.inverse_transform(predicted_volume)

# Part 3

# Visualising the results
plt.plot(real_volume, color = 'red', label = 'Real volume in Lane-2')
plt.plot(predicted_volume, color = 'blue', label = 'Predicted volume in Lane-2')
plt.title('Volume in Lane-2 prediction')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()