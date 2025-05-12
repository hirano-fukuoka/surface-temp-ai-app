import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

class TempDatasetV2(Dataset):
    def __init__(self, df, feature_cols, target_col, window_size=50, depth_col="depth"):
        self.X = []
        self.y = []
        for i in range(len(df) - window_size):
            x_window = df.iloc[i:i+window_size][feature_cols].values
            depth = df.iloc[i+window_size][depth_col]
            depth_column = np.full((window_size, 1), depth)
            x_augmented = np.concatenate([x_window, depth_column], axis=1)
            y_target = df.iloc[i+window_size][target_col]
            self.X.append(x_augmented)
            self.y.append(y_target)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TempPredictorV2(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
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
