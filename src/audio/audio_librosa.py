import librosa
import librosa.display
import numpy as np

def load_audio(path, sr):
    data, sample_rate = librosa.load(path, sr=sr)
    return data, sample_rate

def to_mel_spectrogram(data, sr, n_fft=800, hop_length=320, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    return librosa.power_to_db(mel_spectrogram, ref=np.max)