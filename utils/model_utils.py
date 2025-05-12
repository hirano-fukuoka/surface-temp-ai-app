import os
import torch

def load_latest_model(model_class, model_dir="model/"):
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".pt")],
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("モデルが見つかりません。再学習してください。")
    latest_path = os.path.join(model_dir, model_files[0])
    model = model_class()
    model.load_state_dict(torch.load(latest_path, map_location="cpu"))
    model.eval()
    return model
