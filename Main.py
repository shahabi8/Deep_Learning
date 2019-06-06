import numpy as np
import pandas_datareader as web
from pandas import concat
from sklearn.metrics import mean_squared_error
from Persistance_Algorithm import Persistance_Alg
import datetime
from LSTM import LSTM_Alg 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.preprocessing import MinMaxScaler
import Data_preprocess as DPrep


start = datetime.datetime(2004,1,1)
end = datetime.date.today()
Google_data = web.DataReader("GOOG", "yahoo", start, end)
Adj_Close = Google_data["Adj Close"]
del Google_data['Close']
del Google_data['Low']
del Google_data['High']
del Google_data['Volume']

Persistance_model = Persistance_Alg(Adj_Close)
LSTM_model = LSTM_Alg(Google_data)

fig = plt.figure()
ax = fig.add_subplot(111)
LSTM_model.data['prediction'].plot(grid = True, label = 'LSTM')
LSTM_model.data['Adj Close'].plot(grid = True, label = 'Actual USD Value')
plt.ylabel('Price USD')
plt.xlabel('Date')
ax.set_title('LSTM Prediction')
ax.legend(loc='upper left')
plt.show()
