import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import eeg_to_spectrogram
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
import os
import gdown

# Set the page config before any other Streamlit commands
st.set_page_config(page_title="Brain EEG Classifier", layout="wide")

# 🌟 Define FixedDropout to handle custom layer issue
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        return super(FixedDropout, self).call(inputs, training)

# 👇 Register FixedDropout globally
get_custom_objects().update({'FixedDropout': FixedDropout})

# 🌟 Load models function with Google Drive integration
@st.cache_resource
def load_models():
    models = []
    file_ids = [
        '19vagTsjJushCJ25YikZzkCTyaLFfmfO-',  # Fold 0
        '1LhptLaTjdDQ7KAoKzYCgUqNrvDFdOyci',  # Fold 1
        '1iYXG31bFpLT-eIIFCk7qLSKnd67kwUP8',  # Fold 2
        '1e7AEIA2sdJid1T5_HVDfTZz2NzWGYVhZ',  # Fold 3
        '13KoESOQzPG1GwaFD5BBRT-SudBhkMD-k'   # Fold 4
    ]

    os.makedirs("models", exist_ok=True)

    for i, file_id in enumerate(file_ids):
        model_path = f"models/EffNetB0_Fold{i}.h5"
        if not os.path.exists(model_path):
            st.info(f"Downloading model {i + 1}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model {i + 1}: {e}")
                continue

        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'FixedDropout': FixedDropout}
            )
            models.append(model)
        except Exception as e:
            st.error(f"Error loading model {i + 1}: {e}")

    return models

# Add custom style to the app
st.markdown("""
    <style>
        /* Custom styling */
        .stApp {
            background-color: #f0f0f5;
        }

        /* Title and header styles */
        .css-18e3th9 {
            font-size: 3em;
            font-family: 'Arial', sans-serif;
            color: #4B8B3B;
        }

        /* Style for subheader */
        .css-1v0mbdj {
            font-size: 1.5em;
            font-family: 'Arial', sans-serif;
            color: #444;
        }

        /* Styling for the prediction results */
        .prediction-result {
            font-size: 1.2em;
            font-family: 'Arial', sans-serif;
            color: #333;
            font-weight: bold;
        }

        /* Style for diagnosis results */
        .diagnosis-result {
            font-size: 1.3em;
            font-family: 'Arial', sans-serif;
            color: #4B8B3B;
            font-weight: bold;
        }

        /* Buttons styling */
        .css-1d391kg {
            background-color: #4B8B3B;
            color: white;
            padding: 10px 30px;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 5px;
        }

        /* Add spacing between sections */
        .stTextInput, .stFileUploader {
            margin-bottom: 20px;
        }

        /* Style for results list */
        .stMarkdown {
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Hover effects for results */
        .stButton:hover {
            background-color: #388E3C;
        }

        .stRadio:hover {
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Streamlit App Start
st.title("🧠 Harmful Brain Activity Classifier")

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

        # Load models
        models = load_models()

        if models:
            # Make predictions
            preds = [model.predict(x)[0] for model in models]
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

    except Exception as e:
        st.error(f"Error: {e}")
