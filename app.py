import streamlit as st

# Set the page config before any other Streamlit commands
st.set_page_config(page_title="Brain EEG Classifier", layout="wide")

import tensorflow as tf
import numpy as np
import pandas as pd
from utils import eeg_to_spectrogram

# 🌟 Define FixedDropout to handle custom layer issue
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
import os
import gdown

@st.cache_resource
def load_models():
    models = []
    file_ids = [
        'ID_FOR_FOLD0',
        'ID_FOR_FOLD1',
        'ID_FOR_FOLD2',
        'ID_FOR_FOLD3',
        'ID_FOR_FOLD4'
    ]

    os.makedirs("models", exist_ok=True)

    for i, file_id in enumerate(file_ids):
        model_path = f"models/EffNetB0_Fold{i}.h5"
        if not os.path.exists(model_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FixedDropout': FixedDropout}
        )
        models.append(model)

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

# 🌟 Define FixedDropout class
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        return super(FixedDropout, self).call(inputs, training)

# 👇 Register it globally
get_custom_objects().update({'FixedDropout': FixedDropout})

# 🧠 Streamlit App Start
st.title("🧠 Harmful Brain Activity Classifier")

@st.cache_resource
def load_models():
    models = []
    for i in range(5):
        model = tf.keras.models.load_model(
            f'models/EffNetB0_Fold{i}.h5',
            custom_objects={'FixedDropout': FixedDropout}
        )
        models.append(model)
    return models

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

        models = load_models()
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
