import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from train import train_from_csv
from utils.model_utils import load_latest_model
from utils.training_utils import TempPredictor

def predict_surface(model, df, window_size=50):
    preds = []
    with torch.no_grad():
        for i in range(len(df) - window_size):
            x = df.iloc[i:i+window_size][['T_1mm', 'T_5mm', 'T_10mm']].values
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y_pred = model(x_tensor).squeeze().item()
            preds.append(y_pred)
    return preds

# Streamlit アプリ開始
st.title("🌡️ 表面温度推定AIアプリ（モード自動切替）")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
if uploaded_file:
    csv_path = "data/uploaded_experiment.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    df = pd.read_csv(csv_path)
    st.write("📄 データプレビュー", df.head())

    # モード分岐
   if "T_surface" not in df.columns:
    st.warning("⚠ 推論モード：T_surfaceが存在しません")

    model = load_latest_model(TempPredictor)

    # 推論実行
    preds = []
    window_size = 50
    for i in range(len(df) - window_size):
        x = df.iloc[i:i+window_size][['T_1mm', 'T_5mm', 'T_10mm']].values
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_pred = model(x_tensor).squeeze().item()
        preds.append(y_pred)

    # 結果の描画
    st.subheader("🔍 推定された表面温度")
    st.line_chart(preds)

