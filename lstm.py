# -*- coding: utf-8 -*-

# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
dataset = pd.read_csv('/content/drive/My Drive/data/forex/HISTDATA_COM_MT_EURUSD_M12012/DAT_MT_EURUSD_M1_2012.csv', header=None, names=['date','hour','open','high','low','close','volume'])

minimized_ds = dataset.loc[dataset['hour'].isin(['00:00','06:00','12:00','18:00'])]

all_values = minimized_ds.iloc[:,2:3].values

# create the training set and test set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(minimized_ds, test_size=0.05, shuffle=False)
training_set = train_df.iloc[:, 2:3].values
test_set = test_df.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

predict_size = 40

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(predict_size, len(training_set)):
    X_train.append(training_set_scaled[i-predict_size:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

"""# Training"""

X_train.shape[1]

# Part 2 - Building the RNN

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
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 40, batch_size = 20)

"""# Making prediction and visualizing data"""

inputs = np.concatenate((training_set[-(predict_size):],test_set))

inputs = sc.transform(inputs)

real_stock_price = test_set

X_test = []
for i in range(predict_size, len(inputs)):
    X_test.append(inputs[i-predict_size:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real EURUSD Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted EURUSD Price')
plt.title('Forex Price Prediction')
plt.xlabel('Time')
plt.ylabel('EURUSD Price')
plt.legend()
plt.show()

"""# Analysis model performance"""

past_data = all_values[len(all_values)-len(test_set)-1:-1]

predicted_direction = np.sign(predicted_stock_price - past_data)

actual_direction = np.sign(real_stock_price - past_data)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(actual_direction, predicted_direction) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual_direction, predicted_direction))
print('Report : ')
print(classification_report(actual_direction, predicted_direction))
