from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


DATA_PATH = os.environ.get('DATA_PATH', '../data/DAT_MT_EURUSD_M1_2012.csv')
TIME_STEP = 40
BATCH_SIZE = 100


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


def create_structured_training_set(scaled_train_data):
    """
    Creating a data structure with size TIME_STEP and 4 output
    """
    X_train = []
    y_train = []
    for i in range(TIME_STEP, len(scaled_train_data)):
        X_train.append(scaled_train_data[i-TIME_STEP:i, :])
        y_train.append(scaled_train_data[i, :])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping
    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], scaled_train_data.shape[1]))
    return X_train, y_train


def create_structured_test_set(scaled_train_data, scaled_test_data):
    """
    Creating a data structure of size TIME_STEP
    """
    inputs = np.concatenate(
        (scaled_train_data[-(TIME_STEP):], scaled_test_data))
    X_test = []
    for i in range(TIME_STEP, len(inputs)):
        X_test.append(inputs[i-TIME_STEP:i, :])
    X_test = np.array(X_test)
    return X_test


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


def get_price_direction(predict_data, real_data):
    """
    Price direction: increase or decrease
    Return predicted direction and actual direction
    """
    shift_1_data = real_data[:-1, 0]

    predicted_direction = np.sign(predict_data[1:, 0] - shift_1_data)
    actual_direction = np.sign(real_data[1:, 0] - shift_1_data)
    return predicted_direction, actual_direction


def compute_confusion_matrix(actual, predict):
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
