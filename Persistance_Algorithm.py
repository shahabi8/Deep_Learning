import numpy as np
import pandas as pd
from pandas import concat
from sklearn.metrics import mean_squared_error


class Persistance_Alg():

    def __init__(self, Adj_Close):
        self.Adj_Close = Adj_Close
        self.dataframe = pd.concat([self.Adj_Close.shift(1), self.Adj_Close], axis=1)
        self.dataframe.columns = ['t-1', 't+1']

        print(self.dataframe.head(5))

        # split into train and test sets
        self.train_size = int(len(self.dataframe.index) * 0.66)
        self.train, self.test = self.dataframe[1:self.train_size], self.dataframe[self.train_size:]
        self.train_X, self.train_y = self.train['t-1'], self.train['t+1']
        self.test_X, self.test_y = self.test['t-1'], self.test['t+1']
        self.Prediction()

    def model_persistence(self, x):
        return x

    def Prediction(self):
        # walk-forward validation
        self.predictions = list()
        self.dataframe['prediction'] = self.train_y
        for i in range(len(self.test_X.index)):
            self.yhat = self.model_persistence(self.test_X[i])
            self.predictions.append(self.yhat)
            self.dataframe['prediction'][self.train_size+i] = self.yhat

        self.test_score = mean_squared_error(self.test_y, self.predictions)
        print('Test RMSE: %.3f' % self.test_score)


