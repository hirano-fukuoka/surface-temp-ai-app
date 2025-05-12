import torch
from datetime import datetime
import os

def save_model_by_date(model, dir="model/"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    date_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(dir, f"lstm_{date_str}.pt")
    torch.save(model.state_dict(), path)
    return path

def load_latest_model(model_class, model_dir="model/"):
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".pt")],
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("モデルが見つかりません")
    latest_path = os.path.join(model_dir, model_files[0])
    model = model_class()
    model.load_state_dict(torch.load(latest_path, map_location="cpu"))
    model.eval()
    return model
