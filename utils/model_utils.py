import torch
from datetime import datetime

def save_model_by_date(model, dir="model/"):
    date_str = datetime.now().strftime("%Y%m%d")
    path = f"{dir}/lstm_{date_str}.pt"
    torch.save(model.state_dict(), path)
    return path
