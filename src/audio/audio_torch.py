import torch
import torchaudio
import torchaudio.transforms as T

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    return waveform, sr

def to_mel_spectrogram(audio: tuple, sr, n_fft=800, win_length=800, hop_length=320, n_mels=128):
    waveform, org_freq = audio
    if sr != org_freq:
        resample = T.Resample(orig_freq=org_freq, new_freq=sr)
        waveform = resample(waveform)
        org_freq = 32000
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=800,
        win_length=800,
        hop_length=320,
        n_mels=128
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = mel_transform.to(device)
    waveform = waveform.to(device)

    mel_spectrogram = mel_transform(waveform)
    amplitude_to_db = T.AmplitudeToDB(stype="power")
    mel_s_db = amplitude_to_db(mel_spectrogram)

    return mel_s_db
