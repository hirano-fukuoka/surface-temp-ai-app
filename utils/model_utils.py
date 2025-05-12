import torch
from datetime import datetime
import os

def save_model_by_date(model, model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    model_path = os.path.join(model_dir, f"lstm_{date_str}.pt")
    torch.save(model.state_dict(), model_path)
    return model_path

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
