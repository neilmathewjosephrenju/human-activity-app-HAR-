import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load Models
cnn_model = load_model("activity_model_cnn.keras")
rf_model = joblib.load("activity_model_rf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Human Activity Recognition", layout="wide")
st.title("🧠 Human Activity Recognition (CNN + RF Combined Model)")

uploaded_file = st.file_uploader("📂 Upload your test CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    expected_features = rf_model.n_features_in_

    # Fix extra columns issue
    if df.shape[1] > expected_features:
        st.warning(f"⚠️ Dropping {df.shape[1] - expected_features} extra column(s) to match model features.")
        df = df.iloc[:, :expected_features]
    elif df.shape[1] < expected_features:
        st.error(f"❌ Uploaded CSV has fewer columns ({df.shape[1]}) than expected ({expected_features}).")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.checkbox("Select a row to predict"):
        row_index = st.number_input(
            "Enter the row number (0-based index)", 
            min_value=0, max_value=len(df) - 1, value=0
        )

        # Extract sample
        sample = df.iloc[row_index].values.astype('float32').reshape(1, -1)

        # CNN expects (batch, timesteps, 1) → reshape accordingly
        sample_cnn = sample.reshape((1, sample.shape[1], 1))
        pred_cnn = cnn_model.predict(sample_cnn)

        # Random Forest expects 2D input
        pred_rf = rf_model.predict_proba(sample)

        # Combined Prediction (average probabilities)
        final_pred = (pred_cnn + pred_rf) / 2
        final_label_index = np.argmax(final_pred)
        predicted_activity = label_encoder.inverse_transform([final_label_index])[0]

        st.success(f"🎯 Predicted Activity: **{predicted_activity}**")
