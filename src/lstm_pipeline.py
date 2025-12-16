import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def create_sequences(X, y, store_col_index, lookback=30):
    """
    Creates sequences respecting store boundaries.
    X: Scaled Feature Matrix (numpy)
    y: Scaled Target Vector (numpy)
    store_col_index: The column index where 'store_nbr' is located
    """
    Xs, ys = [], []
    
    # We assume X and y are already sorted by Store then Date
    # But to be safe, we iterate through the data
    
    total_len = len(X)
    

    #window and lookback
    for i in range(total_len - lookback):
        window_x = X[i : i + lookback]
        target_y = y[i + lookback]
        
        
        first_store = window_x[0, store_col_index]
        last_store = window_x[-1, store_col_index]
        target_store = X[i + lookback, store_col_index]
        
        if first_store == last_store == target_store:
            Xs.append(window_x)
            ys.append(target_y)
            
    return np.array(Xs), np.array(ys)



#just a class for the dataset itself
class FavoritaDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.FloatTensor(y_seq)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



#another class for the model itself
class ScalableLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(ScalableLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last time step
        return out



#this will be looped in training
def train_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    print(f"Training on {len(train_loader.dataset)} sequences...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model



#model predictions
def predict(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch)
            preds.extend(output.numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
    return np.array(preds), np.array(actuals)