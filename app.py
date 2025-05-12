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
    if "T_surface" in df.columns:
        st.success("✅ 学習モード（教師データあり）")
        if st.button("モデル再学習を実行"):
            model_path, log = train_from_csv(csv_path)
            st.success(f"✅ 学習完了: {model_path}")
            df_log = pd.DataFrame(log)
            st.line_chart(df_log.set_index("epoch")[["loss", "mae", "rmse"]])
    else:
        st.warning("⚠ 推論モード：T_surfaceが存在しません")
        model = load_latest_model(TempPredictor)
        preds = predict_surface(model, df)
        st.subheader("🔍 推定された表面温度")
        st.line_chart(preds)
