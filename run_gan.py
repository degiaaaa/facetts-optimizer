import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.optim import Adam
import librosa
import numpy as np

# Import necessary components from your project
from model.face_tts import FaceTTS
from model.utils import sequence_mask


# -------------------------
# Voice Feature Extraction
# -------------------------
class VoiceFeatureExtractor:
    def __init__(self, _config):
        self.sampling_rate = _config["sample_rate"]
        self.hop_length = _config["hop_len"]
        self.filter_length = _config["n_fft"]
        self.win_length = _config["win_len"]
        self.n_mels = _config["n_mels"]
        self.mel_fmin = _config["f_min"]
        self.mel_fmax = _config["f_max"]
        #self.config = _config

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


# ------------------------------------------
# Spectrogram Discriminator with Aux Losses
# ------------------------------------------
class SpectrogramDiscriminator(LightningModule):
    def __init__(self, in_channels=1, base_channels=32, num_layers=3, lrelu_slope=0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else base_channels * (2 ** (i - 1)),
                    base_channels * (2 ** i),
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            self.layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))

        self.final_conv = nn.Conv2d(base_channels * (2 ** (num_layers - 1)), 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1), feature_maps


# -------------------------
# FaceTTS with SpecDiffGAN Losses
# -------------------------
class FaceTTSWithDiscriminator(FaceTTS):
    def __init__(self, _config):
        super().__init__(_config)
        self.discriminator = SpectrogramDiscriminator()
        self.feature_extractor = VoiceFeatureExtractor(_config)
        self.adv_criterion = nn.MSELoss()
        self.fm_criterion = nn.L1Loss()
        self.recon_criterion = nn.L1Loss()

    def compute_feature_matching_loss(self, real_features, fake_features):
        """
        Feature Matching Loss: Compares real and fake discriminator feature maps.
        Aligns dimensions by cropping or padding to ensure consistency.
        """
        fm_loss = 0
        for real, fake in zip(real_features, fake_features):
            if real.size(-1) != fake.size(-1):
                if real.size(-1) > fake.size(-1):
                    # Crop real to match fake
                    real = real[..., :fake.size(-1)]
                else:
                    # Pad fake to match real
                    padding = real.size(-1) - fake.size(-1)
                    fake = F.pad(fake, (0, padding), mode="constant", value=0)
            # Compute L1 loss for aligned features
            fm_loss += self.fm_criterion(real, fake)
        return fm_loss


    def compute_pitch_loss(self, real_f0, fake_f0):
        """
        Pitch (F0) Loss: Compares real and fake F0 features.
        Handles dimensional mismatches using cropping or padding.
        """
        if real_f0.size(-1) != fake_f0.size(-1):
            if real_f0.size(-1) > fake_f0.size(-1):
                # Crop real F0
                real_f0 = real_f0[..., :fake_f0.size(-1)]
            else:
                # Pad fake F0
                padding = real_f0.size(-1) - fake_f0.size(-1)
                fake_f0 = F.pad(fake_f0, (0, padding), mode="constant", value=0)
        return self.recon_criterion(real_f0, fake_f0)

    def compute_energy_loss(self, real_energy, fake_energy):
        """
        Energy Loss: Compares real and fake energy features.
        Handles dimensional mismatches using cropping or padding.
        """
        if real_energy.size(-1) != fake_energy.size(-1):
            if real_energy.size(-1) > fake_energy.size(-1):
                # Crop real energy
                real_energy = real_energy[..., :fake_energy.size(-1)]
            else:
                # Pad fake energy
                padding = real_energy.size(-1) - fake_energy.size(-1)
                fake_energy = F.pad(fake_energy, (0, padding), mode="constant", value=0)
        return self.recon_criterion(real_energy, fake_energy)

    def configure_optimizers(self):
        generator_optimizer = Adam(self.parameters(), lr=self.config['learning_rate'])
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.config['learning_rate'])
        return [generator_optimizer, discriminator_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_len, y, y_len, spk = (
            batch['x'].to(self.device),
            batch['x_len'],
            batch['y'].to(self.device),
            batch['y_len'],
            batch['spk'].to(self.device),
        )

        # Generate mel-spectrogram
        encoder_outputs, decoder_outputs, _ = self.forward(x, x_len, self.config['timesteps'], spk=spk)
        generated_mel = decoder_outputs[-1]

        # Extract F0, energy, and mel-spectrogram for real and fake
        real_f0 = self.feature_extractor.extract_f0(y.cpu().numpy())
        real_energy = self.feature_extractor.extract_energy(y.cpu().numpy())
        fake_f0 = self.feature_extractor.extract_f0(generated_mel.detach().cpu().numpy())
        fake_energy = self.feature_extractor.extract_energy(generated_mel.detach().cpu().numpy())

        # Discriminator step
        if optimizer_idx == 1:
            real_logits, real_features = self.discriminator(y.unsqueeze(1))
            fake_logits, fake_features = self.discriminator(generated_mel.detach().unsqueeze(1))

            real_loss = self.adv_criterion(real_logits, torch.ones_like(real_logits))
            fake_loss = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
            return d_loss

        # Generator step
        elif optimizer_idx == 0:
            fake_logits, fake_features = self.discriminator(generated_mel.unsqueeze(1))
            adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))

            # Feature Matching Loss
            real_logits, real_features = self.discriminator(y.unsqueeze(1))
            fm_loss = self.compute_feature_matching_loss(real_features, fake_features)

            # Pitch and Energy Loss
            pitch_loss = self.compute_pitch_loss(real_f0, fake_f0)
            energy_loss = self.compute_energy_loss(real_energy, fake_energy)

            # Combine with existing losses
            dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(x, x_len, y, y_len, spk=spk)
            g_loss = adv_loss + fm_loss + pitch_loss + energy_loss + dur_loss + prior_loss + diff_loss + spk_loss

            self.log("train/duration_loss", dur_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/prior_loss", prior_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/diffusion_loss", diff_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/spk_loss", spk_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/pitch_loss", pitch_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/energy_loss", energy_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
            return g_loss

import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

import os
import copy

from config import ex
from model.face_tts import FaceTTS
from data import _datamodules
import torch

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)

    #os.makedirs(_config["local_checkpoint_dir"], exist_ok=True)

    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        save_weights_only=False,
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    #model = FaceTTS(_config)
    model = FaceTTSWithDiscriminator(_config)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
         _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
     )
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # Load the model's state dictionary -new
    checkpoint = torch.load(_config["resume_from"])
    generator_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if "discriminator" not in k}
    model.load_state_dict(generator_state_dict, strict=False)


    trainer = pl.Trainer(
        #gpus=_config["num_gpus"],
        accelerator="gpu", 
        devices= _config["num_gpus"],#torch.cuda.device_count(), #_config["num_gpus"], #4
        num_nodes=_config["num_nodes"],
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        #flush_logs_every_n_steps=50,
        #weights_summary="top",
        enable_model_summary=True,
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm) #, ckpt_path=_config["resume_from"]
    else:
        trainer.test(model, datamodule=dm) #, ckpt_path=_config["resume_from"]