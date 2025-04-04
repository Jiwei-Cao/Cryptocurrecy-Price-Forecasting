import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Opening and handling the data
data = pd.read_csv('Bitcoin_04_04_2024-04_04_2025_historical_data_coinmarketcap.csv', sep=';')
data = data[['timeOpen', 'close']]
data['timeOpen'] = pd.to_datetime(data['timeOpen'])
#plt.plot(data['timeOpen'], data['close'])

def prepare_dataframe_for_lstm(df, n_steps):
  # Create a deepcopy of the dataframe
  df = dc(df)

  # Set the index of the dataframe to the date
  df.set_index('timeOpen', inplace=True)

  # Shifts the dataframe for the number of lookback windows (n_steps)
  for i in range(1, n_steps+1):
    df[f'Close(t-{i})'] = df['close'].shift(i)

  # Removes any missing values from the dataframe such as NaN
  df.dropna(inplace=True)

  return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
# shifted_df

shifted_df_as_np = shifted_df.to_numpy()
# shifted_df_as_np

# Scale the matrix so that the features all in between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
# shifted_df_as_np

# Split into input features (X) and prediction target (y)
X = shifted_df_as_np[:, 1:]  # All columns except the first (past data)
y = shifted_df_as_np[:, 0]   # First column (the value to predict)
# X.shape, y.shape

# Flip features so LSTM sees the oldest data first (past → present), allowing it to build context over time
X = dc(np.flip(X , axis=1))
# X

# Splitting the data into train and test data
split_index = int(len(X) * 0.95)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

test_dates = shifted_df.index[split_index:]

# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Reshape data to match LSTM input shape: (batch_size, sequence_length, num_features)
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

# Reshape target to 2D array: (batch_size, 1)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# X_train.shape, X_test.shape, y_train.shape, y_test.shape
