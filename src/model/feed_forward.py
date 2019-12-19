from base import BaseModel
import helper
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib

DATA_PATH = os.environ.get('DATA_PATH', '../../data/DAT_MT_EURUSD_M1_2012.csv')
TIME_STEP = 40
BATCH_SIZE = 100


class MlpModel(BaseModel):
    def create_model(self):
        mlp_model = Sequential()
        mlp_model.add(Flatten(input_shape=(TIME_STEP, 4)))
        mlp_model.add(Dense(units=20, activation='linear'))
        mlp_model.add(Dense(units=40, activation='tanh'))
        mlp_model.add(Dense(units=20, activation='tanh'))
        mlp_model.add(Dense(units=4, activation='linear'))
        mlp_model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
        self.model = mlp_model

    def train_model(self, epochs, batch_size):
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs, batch_size=batch_size, shuffle=False)


def create_mlp_model(X_train, y_train, epoch, batch_size):
    mlp = MlpModel(X_train=X_train, y_train=y_train)
    mlp.create_model()
    mlp.train_model(epoch, BATCH_SIZE)
    return mlp


def main(load_existing=False):
    # create the training set and test set
    training_data, test_data = helper.prepare_data(DATA_PATH)

    # Feature Scaling
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(training_data)

    # Create X_train, y_train
    X_train, y_train = helper.create_structured_training_set(
        scaled_train_data)

    epochs = 300

    if(load_existing):
        mlp_model = joblib.load('mlp_model.pkl')
        mlp = MlpModel(mlp_model)
        mlp.set_training_data(X_train, y_train)
        mlp.train_model(epochs, BATCH_SIZE)
    else:
        mlp = create_mlp_model(X_train, y_train, epochs, BATCH_SIZE)
        # Save model to local file
        mlp.export_model('mlp_model')

    # Prepare test data
    scaled_test_data = scaler.transform(test_data)
    X_test = helper.create_structured_test_set(
        scaled_train_data, scaled_test_data)

    # Predict data
    scaled_predicted_price = mlp.predict(X_test, BATCH_SIZE)
    predicted_price = scaler.inverse_transform(scaled_predicted_price)

    # Visualizing and calculate performance
    helper.visualizing(test_data, predicted_price)
    predicted_direction, actual_direction = helper.get_price_direction(
        predicted_price, test_data)
    helper.compute_confusion_matrix(
        actual_direction, predicted_direction)


if __name__ == '__main__':
    if(sys.argv[1]):
        main(load_existing=True)
    else:
        main()
    print('sucessfully trained model')
