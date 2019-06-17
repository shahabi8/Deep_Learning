# Predicting Stock Price Using Long Short-Term Memory Neural Networks
The goal of this project is to predict the future closing price of a stock given its previous closing prices for certain period.
Therefore, this is a regression problem. There are lots of complexities associated with stock prediction since the physical factors vs. physiological, rational and irrational behaviors are all involved and affect the market trend. LSTM is widely used for time series prediction and it’s seen to be very effective. There are a number of papers [7, 8] that suggest Long Short-Term Memory neural networks for time series data like stock price.
I’m planning to investigate LSTM neural net and its implementation using Keras library. 

# Instruction

Main.py: The main file grabs data from Yahoo Finance and pass the data to other classes.
Data_preprocess.py: Containes few functions to preprocess data like normalization, transform data to time series, and transform stock price to price change of stock.
Persistence_Algorithm.py: Implementation of Persistence algorithm.
LSTM.py: Implementation of LSTM neural networks.

# Used Libraries
- Keras
- sklearn
- pandas_datareader
- datetime

# Additional Document

- Murtaza Roondiwala1, Harshal Patel, Shraddha Varma, Predicting Stock Prices Using LSTM. International Journal of Science and Research. Volume 6 Issue 4, April 2017
