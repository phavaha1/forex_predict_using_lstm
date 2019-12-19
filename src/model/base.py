from abc import ABC, abstractmethod
import joblib


class BaseModel(ABC):
    def __init__(self, model=None, X_train=None, y_train=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def set_model(self, model):
        self.model = model

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test_data(self, X_test):
        self.X_test = X_test

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    def export_model(self, name):
        joblib.dump(self.model, name + '.pkl')

    def predict(self, X_test, batch_size):
        predicted_stock_price = self.model.predict(
            X_test, batch_size=batch_size)
        return predicted_stock_price
