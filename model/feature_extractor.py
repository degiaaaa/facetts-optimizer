import torch
import librosa
import numpy as np

class VoiceFeatureExtractor:
    def __init__(self, _config):
        self.sampling_rate = _config["sample_rate"]
        self.hop_length = _config["hop_len"]
        self.filter_length = _config["n_fft"]
        self.win_length = _config["win_len"]
        self.n_mels = _config["n_mels"]
        self.mel_fmin = _config["f_min"]
        self.mel_fmax = _config["f_max"]

    def extract_mel_spectrogram(self, wav):
        """Extract mel-spectrogram using librosa."""
        stft = librosa.stft(
            wav,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel_filter = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )
        mel_spectrogram = np.dot(mel_filter, np.abs(stft))
        return torch.tensor(mel_spectrogram, dtype=torch.float32)

    def extract_f0(self, wav):
        """Extract F0 using librosa's PYIN."""
        f0, _, _ = librosa.pyin(
            wav,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.sampling_rate,
        )
        return torch.tensor(f0, dtype=torch.float32).unsqueeze(0)

    def extract_energy(self, wav):
        """Extract energy (RMS) using librosa."""
        energy = librosa.feature.rms(y=wav, frame_length=self.filter_length, hop_length=self.hop_length)
        return torch.tensor(energy, dtype=torch.float32).squeeze(0)
