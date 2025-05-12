import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

class TempDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window_size=50):
        self.X = []
        self.y = []
        for i in range(len(df) - window_size):
            x_window = df.iloc[i:i+window_size][feature_cols].values
            y_target = df.iloc[i+window_size][target_col]
            self.X.append(x_window)
            self.y.append(y_target)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TempPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def compute_metrics(y_pred, y_true):
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return mae, rmse
