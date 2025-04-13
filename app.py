import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from utils import eeg_to_spectrogram

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

# Define custom Lambda operation for SlicingOpLambda
def slicing_op_lambda(x):
    return x  # Define the actual operation that 'SlicingOpLambda' should perform

# Register the custom operation globally
get_custom_objects().update({'SlicingOpLambda': Lambda(slicing_op_lambda)})

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
            gdown.download(url, model_path, quiet=False)

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FixedDropout': FixedDropout, 'SlicingOpLambda': Lambda(slicing_op_lambda)}
        )
        models.append(model)

    return models

# Your Streamlit app code continues...
