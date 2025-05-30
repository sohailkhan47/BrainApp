import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from utils import eeg_to_spectrogram

# Set the page config before any other Streamlit commands
st.set_page_config(page_title="Brain EEG Classifier", layout="wide")

# 🌟 Streamlit App Start
st.title("🧠 Harmful Brain Activity Classifier")

# Define the function to load models from Google Drive
@st.cache_resource
def load_model(model_url, model_name):
    try:
        # Download the model from Google Drive
        output_path = model_name
        gdown.download(model_url, output_path, quiet=False)
        st.write(f"Model {model_name} downloaded successfully!")

        # Load the model without custom layers for testing purposes
        model = tf.keras.models.load_model(output_path)
        
        # Clean up the downloaded model file after loading
        os.remove(output_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# URLs of models from Google Drive
model_urls = [
    'https://drive.google.com/uc?export=download&id=19vagTsjJushCJ25YikZzkCTyaLFfmfO-',
    'https://drive.google.com/uc?export=download&id=1LhptLaTjdDQ7KAoKzYCgUqNrvDFdOyci',
    'https://drive.google.com/uc?export=download&id=1iYXG31bFpLT-eIIFCk7qLSKnd67kwUP8',
    'https://drive.google.com/uc?export=download&id=1e7AEIA2sdJid1T5_HVDfTZz2NzWGYVhZ',
    'https://drive.google.com/uc?export=download&id=13KoESOQzPG1GwaFD5BBRT-SudBhkMD-k'
]

# Load all models at once
models = [load_model(url, f"EffNetB0_Fold{i}.h5") for i, url in enumerate(model_urls)]

# User uploads the EEG file
uploaded_file = st.file_uploader("📁 Upload EEG `.parquet` File", type=["parquet"])

if uploaded_file:
    st.success("EEG file uploaded. Processing...")

    try:
        df = pd.read_parquet(uploaded_file)
        st.write("📋 EEG Columns Found:", df.columns.tolist())  # Helpful debug

        # Generate the spectrogram
        spec = eeg_to_spectrogram(df)

        # Prepare input for the model
        x = np.zeros((1, 128, 256, 8), dtype='float32')
        for i in range(4):
            x[0,:,:,i] = spec[:,:,i]
            x[0,:,:,i+4] = spec[:,:,i]

        if models:
            preds = [model.predict(x)[0] for model in models if model is not None]
            if preds:
                final_pred = np.mean(preds, axis=0)

                labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

                # Find the label with the highest probability
                max_prob_index = np.argmax(final_pred)
                max_prob_label = labels[max_prob_index]
                max_prob_value = final_pred[max_prob_index]

                # Display the results
                st.subheader("📊 Predicted Probabilities:")

                # Create two columns for displaying results
                result_columns = st.columns(2)
                for i, label in enumerate(labels):
                    prob = final_pred[i]
                    with result_columns[i % 2]:
                        color = "green" if label == max_prob_label else "gray"
                        st.markdown(f"<div style='color: {color}; font-weight: bold;'><b>{label}</b>: {prob:.4f}</div>", unsafe_allow_html=True)

                st.subheader("📝 Diagnosis Result:")
                st.markdown(f"<div class='diagnosis-result'>Highest Probability Diagnosis: {max_prob_label}</div>", unsafe_allow_html=True)
                st.markdown(f"*Probability*: {max_prob_value:.4f}")
                st.markdown("💡 This indicates the most likely harmful brain activity identified in the EEG data.")
            else:
                st.error("No valid predictions from models.")
        else:
            st.error("No models were loaded successfully.")
            
    except Exception as e:
        st.error(f"Error: {e}")
