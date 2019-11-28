# -*- coding: utf-8 -*-
"""forex_price_predict_using_keras_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xsZZ3Lt2W_w3Mj2bD7d7zzJ2TMrya2Qc

# Data Processing
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib


PATH = '/content/drive/My Drive/data/forex/HISTDATA_COM_MT_EURUSD_M12012/DAT_MT_EURUSD_M1_2012.csv'
TIME_STEP = 40


def prepare_data(path):
    dataset = pd.read_csv(path,
                          header=None, names=['date', 'hour', 'open', 'high', 'low', 'close', 'volume'])
    dataset = dataset.fillna(method='ffill')
    minimized_ds = dataset.loc[dataset['hour'].str[-2:].isin(['00'])]
    train_df, test_df = train_test_split(
        minimized_ds, test_size=0.05, shuffle=False)
    training_data = train_df.iloc[:, 2:6].values
    test_data = test_df.iloc[:, 2:6].values
    return training_data, test_data


class LstmModel:
    def __init__(self, training_data):
        self.training_data = training_data
        self.model = self.create_lstm_model()
        self.sc = MinMaxScaler(feature_range=(0, 1))

    def set_training_data(self, training_data):
        self.training_data = training_data

    def set_test_data(self, test_data):
        self.test_data = test_data

    def create_structured_training_set(self):
        """
        Creating a data structure with timesteps and 1 output
        """
        scaled_training_data = self.sc.fit_transform(self.training_data)
        X_train = []
        y_train = []
        for i in range(TIME_STEP, len(scaled_training_data)):
            X_train.append(scaled_training_data[i-TIME_STEP:i, :])
            y_train.append(scaled_training_data[i, :])
        X_train, y_train = np.array(X_train), np.array(y_train)
        # Reshaping
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
        return X_train, y_train

    def create_structured_test_set(self):
        inputs = np.concatenate(
            (self.training_data[-(TIME_STEP):], self.test_data))
        scaled_inputs = self.sc.transform(inputs)
        X_test = []
        for i in range(TIME_STEP, len(scaled_inputs)):
            X_test.append(scaled_inputs[i-TIME_STEP:i, :])
        X_test = np.array(X_test)
        return X_test

    def create_lstm_model(self):
        # Initialising the RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True,
                           input_shape=(TIME_STEP, 4)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units=4))

        # compile model
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        return regressor

    def train_model(self, epochs, batch_size):
        X_train, y_train = self.create_structured_training_set()
        self.model.fit(X_train, y_train,
                       epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.sc.inverse_transform(
            predicted_stock_price)
        return predicted_stock_price


def create_and_train_model(training_data, test_data):
    lstm = LstmModel(training_data, test_data)
    lstm.train_model(40, 20)
    return lstm


def visualizing(real_price, predicted_price):
    # Visualising the results
    plt.figure(figsize=(10, 10))
    # Open bid
    plt.subplot(221)
    plt.plot(real_price[:, 0], color='red', label='Real Open')
    plt.plot(predicted_price[:, 0], color='blue', label='Predicted Open')
    plt.title('Open Bid')
    plt.legend()

    # Close bid
    plt.subplot(222)
    plt.plot(real_price[:, 3], color='red', label='Real Close')
    plt.plot(predicted_price[:, 3],
             color='blue', label='Predicted Close')
    plt.title('Close Bid')
    plt.legend()

    # High bid
    plt.subplot(223)
    plt.plot(real_price[:, 1], color='red', label='Real High')
    plt.plot(predicted_price[:, 1], color='blue', label='Predicted High')
    plt.title('High Bid')
    plt.legend()

    # Low bid
    plt.subplot(224)
    plt.plot(real_price[:, 2], color='red', label='Real Low')
    plt.plot(predicted_price[:, 2], color='blue', label='Predicted Low')
    plt.title('Low Bid')
    plt.legend()

    plt.show()


"""# Analysis model performance"""


def get_price_direction(all_data, predict_data, real_data):
    """
    Price direction: increase or decrease
    Return predicted direction and actual direction
    """
    shift_1_data = all_data[len(all_data)-len(real_data)-1:-1, 0]

    predicted_direction = np.sign(predict_data[:, 0] - shift_1_data)
    actual_direction = np.sign(real_data[:, 0] - shift_1_data)
    return predicted_direction, actual_direction


def analysis_performance(actual, predict):
    """
    Compute the confusion matrix for the predicted price direction and actual price direction
    """
    # Calculate confusion matrix creation.
    results = confusion_matrix(actual, predict)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :', accuracy_score(
        actual, predict))
    print('Report : ')
    print(classification_report(actual, predict))


def main():
    # create the training set and test set
    training_data, test_data = prepare_data(PATH)

    # Feature Scaling
    lstm = create_and_train_model(training_data, test_data)
    joblib.dump(lstm, 'lstm_model.pkl', compress=9)


if __name__ == '__main__':
    main()
