import numpy as np
import pandas as pd
import librosa
import pywt

# Channel names and feature sets
NAMES = ['LL','LP','RP','RR']
FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff[1:]]
    ret = pywt.waverec(coeff, wavelet, mode='per')
    return ret

def eeg_to_spectrogram(df, use_wavelet=False):
    # Get only numeric EEG channels
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 8:
        raise ValueError("Not enough numeric EEG channels found. Need at least 8.")

    # Pick the first 8 channels
    channels = numeric_cols[:8]

    # Create 4 channel-pairs: CH1–CH2, CH3–CH4, CH5–CH6, CH7–CH8
    img = np.zeros((128, 256, 4), dtype='float32')
    for k in range(4):
        ch1 = df[channels[k * 2]].values
        ch2 = df[channels[k * 2 + 1]].values

        x = ch1 - ch2

        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        if use_wavelet:
            x = denoise(x)

        mel_spec = librosa.feature.melspectrogram(
            y=x, sr=200, hop_length=len(x)//256,
            n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
        width = (mel_spec_db.shape[1] // 32) * 32
        mel_spec_db = mel_spec_db[:, :width]
        mel_spec_db = (mel_spec_db + 40) / 40  # Normalize to [0, 1]

        img[:, :, k] = mel_spec_db

    return img

