import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from train import train_from_csv
from utils.training_utils import TempPredictorV2

st.title("🌡️ 任意位置温度推定AIアプリ（修正済み）")

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
        depth_mm = st.number_input("推定したい深さ（mm）を入力", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

        if selected_model:
            model_path = os.path.join(model_dir, selected_model)
            model = TempPredictorV2()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            def inverse_scale(val_norm, min_temp=25.0, max_temp=100.0):
                return val_norm * (max_temp - min_temp) + min_temp

            preds_surface, preds_custom = [], []
            window_size = 50
            for i in range(len(df) - window_size):
                x = df.iloc[i:i+window_size][['T_1mm', 'T_5mm', 'T_10mm']].values
                x_scaled = (x - 25.0) / (100.0 - 25.0)

                depth_feature = np.full((window_size, 1), depth_mm / 10.0)
                surface_feature = np.full((window_size, 1), 0.0)

                x_custom = np.concatenate([x_scaled, depth_feature], axis=1)
                x_surface = np.concatenate([x_scaled, surface_feature], axis=1)

                if x_custom.shape != (window_size, 4) or np.any(np.isnan(x_custom)):
                    continue

                x_tensor_custom = torch.tensor(x_custom, dtype=torch.float32).unsqueeze(0)
                x_tensor_surface = torch.tensor(x_surface, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    y_pred_custom = inverse_scale(model(x_tensor_custom).squeeze().item())
                    y_pred_surface = inverse_scale(model(x_tensor_surface).squeeze().item())

                preds_custom.append(y_pred_custom)
                preds_surface.append(y_pred_surface)

            df_result = pd.DataFrame({
                'time': df['time'].iloc[window_size:].values,
                f'predicted_T_{depth_mm:.1f}mm': preds_custom,
                'predicted_T_surface': preds_surface,
                'T_1mm': df['T_1mm'].iloc[window_size:].values,
                'T_5mm': df['T_5mm'].iloc[window_size:].values,
                'T_10mm': df['T_10mm'].iloc[window_size:].values,
            })

            st.subheader("📈 センサ応答 + 推定された温度")
            st.line_chart(df_result.set_index("time"))

            csv_file = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 推定結果をCSVでダウンロード",
                data=csv_file,
                file_name="predicted_temperature.csv",
                mime="text/csv"
            )
