import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils.training_utils import TempDatasetV2, TempPredictorV2, compute_metrics
from utils.model_utils import save_model_by_date

def train_from_csv(csv_path, window_size=50, num_epochs=20, lr=0.001):
    df = pd.read_csv(csv_path)
    feature_cols = ['T_1mm', 'T_5mm', 'T_10mm']
    target_col = 'T_surface'

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    df[feature_cols] = x_scaler.fit_transform(df[feature_cols])
    df[target_col] = y_scaler.fit_transform(df[[target_col]])
    df["depth"] = 0.0

    dataset = TempDatasetV2(df, feature_cols, target_col, window_size, "depth")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TempPredictorV2(input_size=4, hidden_size=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses, maes, rmses = [], [], []
        for x, y_true in loader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae, rmse = compute_metrics(y_pred, y_true)
            losses.append(loss.item())
            maes.append(mae)
            rmses.append(rmse)

        log.append({
            "epoch": epoch,
            "loss": sum(losses)/len(losses),
            "mae": sum(maes)/len(maes),
            "rmse": sum(rmses)/len(rmses)
        })

    model_path = save_model_by_date(model)
    return model_path, log
