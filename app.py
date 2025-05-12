import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from train import train_from_csv
from utils.training_utils import TempPredictor

st.title("🌡️ 表面温度推定AIアプリ（モデル選択・逆スケーリング対応）")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
if uploaded_file:
    csv_path = "data/uploaded_experiment.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    df = pd.read_csv(csv_path)
    st.write("📄 データプレビュー", df.head())

    if "T_surface" in df.columns:
        st.success("✅ 学習モード（教師データあり）")
        if st.button("モデル再学習を実行"):
            model_path, log = train_from_csv(csv_path)
            st.success(f"✅ 学習完了: {model_path}")
            df_log = pd.DataFrame(log)
            st.line_chart(df_log.set_index("epoch")[["loss", "mae", "rmse"]])
    else:
        st.warning("⚠ 推論モード：T_surfaceが存在しません")

        model_dir = "model"
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".pt")],
            reverse=True
        )
        selected_model = st.selectbox("使用するモデルを選択してください：", model_files)

        if selected_model:
            model_path = os.path.join(model_dir, selected_model)
            model = TempPredictor()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # 逆スケーリング関数（MinMax 25〜100）
            def inverse_scale(val_norm, min_temp=25.0, max_temp=100.0):
                return val_norm * (max_temp - min_temp) + min_temp

            preds = []
            window_size = 50
            for i in range(len(df) - window_size):
                x = df.iloc[i:i+window_size][['T_1mm', 'T_5mm', 'T_10mm']].values
                x_scaled = (x - 25.0) / (100.0 - 25.0)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
                y_pred_norm = model(x_tensor).squeeze().item()
                y_pred_actual = inverse_scale(y_pred_norm)
                preds.append(y_pred_actual)

            df_result = pd.DataFrame({
                'time': df['time'].iloc[window_size:].values,
                'predicted_T_surface': preds,
                'T_1mm': df['T_1mm'].iloc[window_size:].values,
                'T_5mm': df['T_5mm'].iloc[window_size:].values,
                'T_10mm': df['T_10mm'].iloc[window_size:].values,
            })
            st.subheader("📈 センサ応答 + 推定された表面温度")
            st.line_chart(df_result.set_index("time"))

            # 👇 ここに追加
            csv_download = df_result.copy()
            csv_download.reset_index(inplace=True)
            csv_file = csv_download.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 推定結果をCSVでダウンロード",
                data=csv_file,
                file_name="predicted_surface_temperature.csv",
                mime="text/csv"
            )
