import os
import torch
import torchaudio
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from tqdm import tqdm
from scipy.spatial.distance import cosine
from torchaudio.transforms import MelSpectrogram
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import parselmouth
import matplotlib.pyplot as plt

from model.face_tts import FaceTTS
from model.syncnet_hifigan import SyncNet
from model.discriminator import SpectrogramDiscriminator
from utils.mel_spectrogram import mel_spectrogram
from config import ex

@ex.automain
def main(_config):
    pl.seed_everything(_config["seed"])

    print("\n######## Initializing Models ########")
    tts_model = FaceTTS(_config).cuda().eval()
    syncnet = SyncNet(_config).cuda().eval()
    
    use_gan = _config["use_gan"]
    discriminator = None
    if use_gan:
        print("######## Using GAN Architecture ########")
        discriminator = SpectrogramDiscriminator(_config).cuda().eval()
    else:
        print("######## Using Standard FaceTTS ########")

    # Evaluation parameters
    sample_rate = _config["sample_rate"]
    num_mels = _config["n_mels"]
    n_fft = _config["n_fft"]
    hop_length = _config["hop_len"]
    win_length = _config["win_len"]
    f_min = _config["f_min"]
    f_max = _config["f_max"]

    # Paths for evaluation data
    # generated_audio_dir = "/mnt/qb/work/butz/bst080/faceGANtts/test/synth_voices_gan"
    # reference_audio_dir = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/wav/test"
    generated_audio_dir = _config.get("output_dir_gan")
    reference_audio_dir = _config.get("output_dir_orig")
    
    # Create output directory for plots
    plot_dir = "eval_plots"
    os.makedirs(plot_dir, exist_ok=True)

    def find_wav_files(root_dir):
        return sorted([os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if f.endswith(".wav")])

    generated_wavs = find_wav_files(generated_audio_dir)
    reference_wavs = find_wav_files(reference_audio_dir)

    assert len(generated_wavs) == len(reference_wavs), "Mismatch in number of generated and reference audio files!"

    # Metrics containers
    speaker_similarities, feature_matching_losses = [], []
    l1_spectrogram_losses, f0_errors, mfcc_distances = [], [], []
    mcd_values, stft_distances = [], []

    with torch.no_grad():
        for idx, (gen_wav, ref_wav) in tqdm(enumerate(zip(generated_wavs, reference_wavs)), total=len(generated_wavs), desc="Evaluating"):

            print(f"\n[Processing Pair {idx+1}/{len(generated_wavs)}]")
            print(f"Generated: {gen_wav}")
            print(f"Reference: {ref_wav}")

            # Load audio
            gen_audio, sr_gen = torchaudio.load(gen_wav)
            ref_audio, sr_ref = torchaudio.load(ref_wav)

            # Resampling if necessary
            if sr_gen != sample_rate:
                gen_audio = torchaudio.transforms.Resample(orig_freq=sr_gen, new_freq=sample_rate)(gen_audio)
            if sr_ref != sample_rate:
                ref_audio = torchaudio.transforms.Resample(orig_freq=sr_ref, new_freq=sample_rate)(ref_audio)

            # Compute Mel Spectrograms
            mel_gen = mel_spectrogram(gen_audio, n_fft, num_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False)
            mel_ref = mel_spectrogram(ref_audio, n_fft, num_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False)

            print(f"Generated Mel Shape: {mel_gen.shape}, Reference Mel Shape: {mel_ref.shape}")

            # Ensure matching time dimension by truncation
            min_length = min(mel_gen.shape[2], mel_ref.shape[2])
            mel_gen = mel_gen[:, :, :min_length]
            mel_ref = mel_ref[:, :, :min_length]

            print(f"Generated Mel Shape: {mel_gen.shape}, Reference Mel Shape: {mel_ref.shape}")

            # Speaker Similarity using SyncNet
            spk_emb_gen = syncnet.forward_aud(mel_gen.unsqueeze(1).cuda())
            spk_emb_ref = syncnet.forward_aud(mel_ref.unsqueeze(1).cuda())
            spk_emb_gen = spk_emb_gen.mean(dim=-1).squeeze()
            spk_emb_ref = spk_emb_ref.mean(dim=-1).squeeze()
            speaker_similarity = 1 - cosine(spk_emb_gen.cpu().numpy(), spk_emb_ref.cpu().numpy())
            speaker_similarities.append(speaker_similarity)

            # Feature Matching Loss (GAN)
            if use_gan:
                _, real_features = discriminator(mel_ref.unsqueeze(1).cuda())
                _, fake_features = discriminator(mel_gen.unsqueeze(1).cuda())
                fm_loss = sum(F.l1_loss(r, f) for r, f in zip(real_features, fake_features)).item()
                feature_matching_losses.append(fm_loss)

            # Spectrogram L1 Loss
            l1_loss = F.l1_loss(mel_gen, mel_ref).item()
            l1_spectrogram_losses.append(l1_loss)

            # F0 Error Analysis
            snd_gen, snd_ref = parselmouth.Sound(gen_wav), parselmouth.Sound(ref_wav)
            f0_gen, f0_ref = snd_gen.to_pitch().selected_array['frequency'], snd_ref.to_pitch().selected_array['frequency']

            min_length = min(len(f0_gen), len(f0_ref))
            f0_error = np.mean(np.abs(f0_gen[:min_length] - f0_ref[:min_length]))
            f0_errors.append(f0_error)

            # Plot F0 curves for debugging
            # plt.figure(figsize=(10, 4))
            # plt.plot(f0_gen, label="Generated F0")
            # plt.plot(f0_ref, label="Reference F0")
            # plt.legend()
            # plt.title(f"F0 Comparison for Sample {idx}")
            # plt.savefig(os.path.join(plot_dir, f"f0_{idx}.png"))
            # plt.close()

            # MFCC Distance (DTW)
            mfcc_gen = torchaudio.functional.compute_deltas(mel_gen.mean(dim=0))
            mfcc_ref = torchaudio.functional.compute_deltas(mel_ref.mean(dim=0))
            mfcc_dist, _ = fastdtw(mfcc_gen.cpu().numpy(), mfcc_ref.cpu().numpy(), dist=euclidean)
            mfcc_distances.append(mfcc_dist)

            # Mel Cepstral Distortion (MCD)
            mcd = np.mean(np.abs(mel_gen.cpu().numpy() - mel_ref.cpu().numpy()))
            mcd_values.append(mcd)

            # STFT Distance
            stft_dist = torch.norm(mel_gen - mel_ref, p='fro').item()
            stft_distances.append(stft_dist)
            
    # Speichere Debug-Plots (wie bereits vorhanden)
    plt.figure()
    plt.hist(f0_errors, bins=30, alpha=0.7, label="F0 Errors")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "f0_error_histogram.png"))
    plt.close()

    plt.figure()
    plt.hist(stft_distances, bins=30, alpha=0.7, label="STFT Distances")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "stft_error_histogram.png"))
    plt.close()

    # Ausgabe der Mittelwerte
    print("\n######## Evaluation Results ########")
    print(f"Mean Speaker Similarity: {np.mean(speaker_similarities):.4f}")
    print(f"Mean F0 Error: {np.mean(f0_errors):.4f}")
    print(f"Mean MCD: {np.mean(mcd_values):.4f}")
    print(f"Mean STFT Distance: {np.mean(stft_distances):.4f}")

    # --- Hier den Composite Score berechnen und ausgeben ---
    mean_speaker_similarity = np.mean(speaker_similarities)
    mean_f0_error = np.mean(f0_errors)
    mean_mcd = np.mean(mcd_values)
    mean_stft_distance = np.mean(stft_distances)
    composite = mean_f0_error + mean_mcd + (0.01 * mean_stft_distance) - (10 * mean_speaker_similarity)
    print(f"Composite Metric: {composite:.4f}")