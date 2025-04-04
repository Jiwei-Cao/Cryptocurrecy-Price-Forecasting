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
