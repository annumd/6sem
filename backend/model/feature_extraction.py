import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True, duration=5)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_mean = np.mean(spectral.T, axis=0)

        features = np.hstack([mfcc_mean, chroma_mean, spectral_mean])

        # remove NaN or infinite values
        features = np.nan_to_num(features)

        # ensure fixed size for model
        if len(features) != 59:
            return np.zeros(59)

        return features

    except Exception as e:
        print("Feature extraction error:", e)
        return np.zeros(59)
