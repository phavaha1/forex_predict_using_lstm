from base import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
import os
import helper
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = os.environ.get('DATA_PATH', '../../data/DAT_MT_EURUSD_M1_2012.csv')
TIME_STEP = 40
BATCH_SIZE = 100


class LstmModel(BaseModel):
    def __init__(self, model=None, X_train=None, y_train=None, stateful=False):
        super().__init__(model, X_train, y_train)
        self.stateful = stateful

    def create_model(self, batch_size):
        # Initialising the RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True,
                           batch_size=batch_size,
                           input_shape=(
                               TIME_STEP, self.X_train.shape[2]),
                           stateful=self.stateful))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(
            LSTM(units=50, return_sequences=True, stateful=self.stateful))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(
            LSTM(units=50, return_sequences=True, stateful=self.stateful))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, stateful=self.stateful))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units=self.X_train.shape[2]))

        # compile model
        regressor.compile(optimizer='adam', loss='mean_absolute_error')

        self.model = regressor

    def train_model(self, epochs, batch_size):
        if(self.stateful):
            for i in range(epochs):
                self.model.fit(self.X_train, self.y_train,
                               epochs=1, batch_size=batch_size,
                               shuffle=False)
                self.model.reset_states()
        else:
            self.model.fit(self.X_train, self.y_train,
                           epochs=epochs, batch_size=batch_size,
                           shuffle=False)


def create_lstm_model(X_train, y_train, epoch, batch_size, stateful):
    lstm = LstmModel(X_train=X_train, y_train=y_train, stateful=True)
    lstm.create_model(batch_size)
    lstm.train_model(epoch, batch_size)
    return lstm


def main():
    # create the training set and test set
    training_data, test_data = helper.prepare_data(DATA_PATH)

    # the input len to LSTM network should be dividable by the BATCH_SIZE
    # + TIME_STEP because our data is sequence
    train_len = training_data.shape[0] - \
        (training_data.shape[0] % BATCH_SIZE) - 2*BATCH_SIZE
    training_data = training_data[:(train_len+TIME_STEP), :]
    print(training_data.shape)
    # the test len to LSTM network should be dividable by the BATCH_SIZE
    test_len = test_data.shape[0] - (test_data.shape[0] % BATCH_SIZE)
    test_data = test_data[:test_len, :]

    # Feature Scaling
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(training_data)

    # Create X_train, y_train
    X_train, y_train = helper.create_structured_training_set(
        scaled_train_data)

    lstm = create_lstm_model(X_train, y_train, 40, BATCH_SIZE, True)

    # Save model to local file
    lstm.export_model('lstm_model')

    # Prepare test data
    scaled_test_data = scaler.transform(test_data)
    X_test = helper.create_structured_test_set(
        scaled_train_data, scaled_test_data)

    # Predict data
    scaled_predicted_price = lstm.predict(X_test, BATCH_SIZE)
    predicted_price = scaler.inverse_transform(scaled_predicted_price)

    # Visualizing and calculate performance
    helper.visualizing(test_data, predicted_price)
    predicted_direction, actual_direction = helper.get_price_direction(
        predicted_price, test_data)
    helper.compute_confusion_matrix(
        actual_direction, predicted_direction)


if __name__ == '__main__':
    main()
    print('sucessfully trained model')
