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

# Creating custom dataset
class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Wrap the datasets in dataloaders to get batches of data
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Creating the LSTM model
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    # Define the LSTM layer
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

    # Fully connected layer (fc) to map LSTM output to final prediction
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    batch_size = x.size(0)

    #Initialise hidden state (h0 - short term memory) and cell state (c0 - long term memory) with zeros
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

    # Pass input and initial states into LSTM
    # out: output from all timesteps for each sequence
    # _: tuple containing the final hidden and cell states (not used)
    out, _ = self.lstm(x, (h0, c0))

    # Use the output from the last timestep as the input to the final layer
    # out[:, -1, :] selects the last ouput in the sequence for each sample
    out = self.fc(out[:, -1, :])

    # Return the final prediction
    return out

model = LSTM(1, 16, 1)
model.to(device)
# model

# Training and testing the model
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch():
  model.train()
  print(f"Epoch: {epoch + 1}")
  running_loss = 0.0

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 5 == 4:
      avg_loss_across_batches = running_loss / 100
      # print(f'Batch {batch_index + 1}, Loss: {avg_loss_across_batches:.3f}')
      running_loss = 0.0

def validate_one_epoch():
  model.eval()
  with torch.inference_mode():
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
      x_batch, y_batch = batch[0].to(device), batch[1].to(device)

      output = model(x_batch)
      loss = loss_function(output, y_batch)
      running_loss += loss.item()

  avg_loss_across_batches = running_loss / len(test_loader)

  print(f'Test Loss: {avg_loss_across_batches:.3f}')
  print()

for epoch in range(NUM_EPOCHS):
  train_one_epoch()
  validate_one_epoch()

# Plotting the model
model.eval()
with torch.inference_mode():
  # Get model predictions for the training data
  test_predictions = model(X_test.to(device)).to('cpu').numpy().flatten()

  # Prepare dummy array to inverse transform predictions
  # Needs to match original scaled input shape: (samples, lookback+1)
  dummies = np.zeros((X_test.shape[0], lookback+1))
  dummies[:, 0] = test_predictions # Place predictions in the first column

  # Inverse transform to convert predictions back to original scale (e.g., dollars)
  dummies = scaler.inverse_transform(dummies)
  test_predictions = dc(dummies[:, 0]) # Extract rescaled predictions

  # Repeat the same process for the actual test labels
  dummies = np.zeros((X_test.shape[0], lookback+1))
  dummies[:, 0] = y_test.flatten() # Actual values in the first column
  dummies = scaler.inverse_transform(dummies)
  new_y_test = dc(dummies[:, 0]) # Extract rescaled actual values

  plt.plot(test_dates, new_y_test, label='Actual Close')
  plt.plot(test_dates, test_predictions, label='Predicted Close')
  plt.xticks(rotation=45)
  plt.xlabel('Day')
  plt.ylabel('Close')
  plt.legend()
  plt.show()
