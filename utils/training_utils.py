import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

class TempDataset(Dataset):
    def __init__(self, df, window_size):
        self.inputs, self.labels = [], []
        for i in range(len(df) - window_size):
            x = df.iloc[i:i+window_size][['T_1mm', 'T_5mm', 'T_10mm']].values
            y = df.iloc[i+window_size]['T_surface']
            self.inputs.append(torch.tensor(x, dtype=torch.float32))
            self.labels.append(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class TempPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return mae, rmse
