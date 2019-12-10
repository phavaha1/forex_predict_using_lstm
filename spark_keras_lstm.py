from elephas.spark_model import SparkModel
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from elephas.utils.rdd_utils import to_simple_rdd
import numpy as np
from sklearn.model_selection import train_test_split


predict_size = 40

spark = SparkSession.builder.appName('Spark SQL')\
    .config('hive.metastore.uris', 'thrift://10.121.31.83:9083')\
    .enableHiveSupport()\
    .getOrCreate()


spark.sql('use forex')
df = spark.sql('select * from 2012_m1_eurusd').toPandas()

minimized_ds = df.loc[df['time'].isin(['00:00'])]
train_df, test_df = train_test_split(
    minimized_ds, test_size=0.05, shuffle=False)
training_set = train_df.iloc[:, 2:6].values
test_set = test_df.iloc[:, 2:6].values


# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(predict_size, len(training_set)):
    X_train.append(training_set_scaled[i-predict_size:i, :])
    y_train.append(training_set_scaled[i, :])
X_train, y_train = np.array(X_train), np.array(y_train)

rdd = to_simple_rdd(spark.sparkContext, X_train, y_train)


def create_lstm_model():
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True,
                       input_shape=(predict_size, 4)))
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


model = create_lstm_model()

spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0)
