import streamlit as st
from train import train_from_csv
import pandas as pd
import matplotlib.pyplot as plt

st.title("AIによる表面温度推定：学習＆推定アプリ")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
if uploaded_file:
    csv_path = "data/uploaded_experiment.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ データアップロード完了")

    if st.button("モデル再学習を実行"):
        model_path, log = train_from_csv(csv_path)
        st.success(f"✅ 学習完了：モデル保存先 → {model_path}")

        df_log = pd.DataFrame(log)
        st.line_chart(df_log.set_index("epoch")[["loss", "mae", "rmse"]])
