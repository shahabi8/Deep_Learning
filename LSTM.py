import math
import pandas as pd
import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from Data_preprocess import Data_prep



class LSTM_Alg():
    def __init__(self, data):
        self.data = data
        self.index_open = ['Open']
        self.index_Adj_close = ['Adj Close']
        self.DPrep_Open = Data_prep(self.data)
        self.DPrep_Open.Get_normalised_data(self.index_open)
        self.DPrep_Adj_close = Data_prep(self.data)
        self.DPrep_Adj_close.Get_normalised_data(self.index_Adj_close)
        print(self.data.head())

        self.seq_len = 0
        self.tt_split = 0.7
        self.sequence_length = self.seq_len + 1

        self.result = []
        self.data_np = np.array(self.data)
        self.data_fl = self.data_np.astype(np.float)
        self.bounds = [np.amin(self.data_fl), np.amax(self.data_fl)]

        for j in range(len(self.data) - self.sequence_length + 1):
            self.x = self.data[j: j + self.sequence_length]
            self.result.append(np.array(self.x))


        self.result = np.array(self.result)
        
        self.row = round(self.tt_split * self.result.shape[0])
        self.train = self.result[:int(self.row), :, :]

        self.x_train = self.train[:, :, :-1]
        self.y_train = self.train[:, :, -1]
        self.x_test = self.result[int(self.row):, :, :-1]
        self.y_test = self.result[int(self.row):, :, -1]

        self.layers = []
        self.layers.append(self.x_train.shape[-1])
        self.layers.append(128)
        self.layers.append(64)
        self.layers.append(self.sequence_length)

        self.build_model(self.layers)

        print(len(self.data))

        self.data['prediction'] = self.predictions

        self.DPrep_Open.Get_de_normalised_data(self.index_open)
        self.DPrep_Adj_close.Get_de_normalised_data(self.index_Adj_close)
        self.index_prediction = ['prediction']
        self.DPrep_Adj_close.Get_de_normalised_data(self.index_prediction)

        print(self.data.head())

    def get_normalised_data(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.numerical = self.data.columns
        self.data[self.numerical] = self.scaler.fit_transform(self.data[self.numerical])

        return self.data

    def build_model(self, layers):
        self.layers = layers
        self.batch_size = 1
        self.epochs = 1
        self.model = Sequential()

        self.model.add(LSTM(input_shape=(None, self.layers[0]), units=self.layers[3], return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(self.layers[1], return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(self.layers[2], return_sequences=False))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.layers[3]))
        self.model.add(Activation("linear"))

        self.model.compile(loss="mse", optimizer="rmsprop")
        self.model.fit(self.x_train, 
            self.y_train, 
            nb_epoch=self.epochs, 
            batch_size=self.batch_size,
            verbose=2, 
            validation_split=0.05)
        self.predictions_test = self.model.predict(self.x_test)
        self.predictions_train = self.model.predict(self.x_train)
        self.trainScore = self.model.evaluate(self.x_train, self.y_train, verbose=0)

        print('Train Score: %.8f MSE (%.8f RMSE)' % (self.trainScore, math.sqrt(self.trainScore)))

        self.testScore = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (self.testScore, math.sqrt(self.testScore)))
        self.predictions = np.concatenate([self.predictions_train, self.predictions_test])
