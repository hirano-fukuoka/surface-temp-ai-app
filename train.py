import pandas as pd
import torch
from utils.training_utils import TempDataset, TempPredictor, compute_metrics
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.model_utils import save_model_by_date
from datetime import datetime

def train_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    dataset = TempDataset(df, window_size=50)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TempPredictor(input_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    log = {"epoch": [], "loss": [], "mae": [], "rmse": []}
    for epoch in range(20):
        total_loss, preds, labels = 0, [], []
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = model(xb).squeeze()
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds.extend(y_pred.detach().numpy())
            labels.extend(yb.numpy())

        mae, rmse = compute_metrics(preds, labels)
        log["epoch"].append(epoch + 1)
        log["loss"].append(total_loss)
        log["mae"].append(mae)
        log["rmse"].append(rmse)

    # 保存
    model_path = save_model_by_date(model)
    date_str = datetime.now().strftime("%Y%m%d")
    pd.DataFrame(log).to_csv(f"logs/training_log_{date_str}.csv", index=False)
    return model_path, log
