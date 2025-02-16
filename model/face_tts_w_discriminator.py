# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.cuda.amp as amp
# from torch.optim import Adam
# import pytorch_lightning as pl
# from torch.nn.utils import weight_norm, spectral_norm
# from model.face_tts import FaceTTS
# from model.feature_extractor import VoiceFeatureExtractor
# from model.discriminator import SpectrogramDiscriminator

# # ------------------------------------------
# # FaceTTS with SpecDiffGAN Losses and Microbatching for Both Generator and Discriminator
# # ------------------------------------------
# class FaceTTSWithDiscriminator(FaceTTS):
#     def __init__(self, _config):
#         super().__init__(_config)
#         self.config = _config
#         self.discriminator = SpectrogramDiscriminator(_config)
#         self.feature_extractor = VoiceFeatureExtractor(_config)
#         self.adv_criterion = nn.MSELoss()
#         # Falls benötigt: 
#         # self.fm_criterion = nn.L1Loss()
#         # self.recon_criterion = nn.L1Loss()
        
#         # Automatische Optimierung deaktivieren, da wir manuell (und microbatch-weise) optimieren.
#         self.automatic_optimization = False

#     def compute_feature_matching_loss(self, real_features, fake_features):
#         fm_loss = 0
#         for real, fake in zip(real_features, fake_features):
#             if real.size(-1) != fake.size(-1):
#                 if real.size(-1) > fake.size(-1):
#                     real = real[..., :fake.size(-1)]
#                 else:
#                     padding = real.size(-1) - fake.size(-1)
#                     fake = F.pad(fake, (0, padding), mode="constant", value=0)
#             fm_loss += self.fm_criterion(real, fake)
#         return fm_loss

#     def compute_pitch_loss(self, real_f0, fake_f0):
#         if real_f0.size(-1) != fake_f0.size(-1):
#             if real_f0.size(-1) > fake_f0.size(-1):
#                 real_f0 = real_f0[..., :fake_f0.size(-1)]
#             else:
#                 padding = real_f0.size(-1) - fake_f0.size(-1)
#                 fake_f0 = F.pad(fake_f0, (0, padding), mode="constant", value=0)
#         return self.recon_criterion(real_f0, fake_f0)

#     def compute_energy_loss(self, real_energy, fake_energy):
#         if real_energy.size(-1) != fake_energy.size(-1):
#             if real_energy.size(-1) > fake_energy.size(-1):
#                 real_energy = real_energy[..., :fake_energy.size(-1)]
#             else:
#                 padding = real_energy.size(-1) - fake_energy.size(-1)
#                 fake_energy = F.pad(fake_energy, (0, padding), mode="constant", value=0)
#         return self.recon_criterion(real_energy, fake_energy)

#     def configure_optimizers(self):
#         generator_optimizer = Adam(self.parameters(), lr=self.config['learning_rate'])
#         discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.config['learning_rate'])
#         return [generator_optimizer, discriminator_optimizer]

#     def training_step(self, batch, batch_idx):
#         # Daten auf das Gerät verschieben
#         x = batch['x'].to(self.device)
#         x_len = batch['x_len']
#         y = batch['y'].to(self.device)
#         y_len = batch['y_len']
#         spk = batch['spk'].to(self.device)
#         B = x.shape[0]

#         # ----- DISKRIMINATOR-UPDATE (MICROBATCHED) -----
#         micro_batch_size = self.config.get("micro_batch_size", 16)
#         n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size
#         opt_disc = self.optimizers()[1]
#         opt_disc.zero_grad()
#         total_d_loss = 0.0

#         for i in range(n_micro_batches):
#             start = i * micro_batch_size
#             end = min((i + 1) * micro_batch_size, B)
#             x_mini = x[start:end]
#             x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
#             y_mini = y[start:end]
#             y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
#             spk_mini = spk[start:end]

#             with amp.autocast():
#                 enc_out, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
#                 generated_mel_mini = dec_out[-1]
#                 real_logits, _ = self.discriminator(y_mini.unsqueeze(1))
#                 fake_logits, _ = self.discriminator(generated_mel_mini.detach().unsqueeze(1))
#                 # Falls der Discriminator mehrere Outputs liefert:
#                 real_logits = real_logits if isinstance(real_logits, torch.Tensor) else real_logits[0]
#                 fake_logits = fake_logits if isinstance(fake_logits, torch.Tensor) else fake_logits[0]
#                 loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
#                 loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
#                 micro_loss_d = (loss_real + loss_fake) / 2

#             # Optional: Skalierung, damit nach Akkumulation der Durchschnittswert entspricht
#             micro_loss_d = micro_loss_d / n_micro_batches
#             self.manual_backward(micro_loss_d)
#             total_d_loss += micro_loss_d.item()

#         opt_disc.step()
#         avg_d_loss = total_d_loss  # Bereits als Durchschnitt berechnet

#         # ----- GENERATOR-UPDATE (MICROBATCHED) -----
#         micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
#         n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen
#         opt_gen = self.optimizers()[0]
#         opt_gen.zero_grad()
#         total_g_loss = 0.0

#         for i in range(n_micro_batches_gen):
#             start = i * micro_batch_size_gen
#             end = min((i + 1) * micro_batch_size_gen, B)
#             x_mini = x[start:end]
#             x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
#             y_mini = y[start:end]
#             y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
#             spk_mini = spk[start:end]

#             with amp.autocast():
#                 enc_out, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
#                 generated_mel_mini = dec_out[-1]
#                 fake_logits, _ = self.discriminator(generated_mel_mini.unsqueeze(1))
#                 fake_logits = fake_logits if isinstance(fake_logits, torch.Tensor) else fake_logits[0]
#                 adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
#                 # Zusätzliche Diffusionsverluste
#                 dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini)
#                 micro_loss_g = adv_loss + dur_loss + prior_loss + diff_loss + spk_loss

#             micro_loss_g = micro_loss_g / n_micro_batches_gen
#             self.manual_backward(micro_loss_g)
#             total_g_loss += micro_loss_g.item()

#         opt_gen.step()
#         avg_g_loss = total_g_loss  # Durchschnittlicher Generator-Loss

#         # Logge Schritt- und Epochenwerte separat:
#         self.log("train/d_loss_step", avg_d_loss, prog_bar=True, on_step=True, on_epoch=False)
#         self.log("train/d_loss_epoch", avg_d_loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log("train/g_loss_step", avg_g_loss, prog_bar=True, on_step=True, on_epoch=False)
#         self.log("train/g_loss_epoch", avg_g_loss, prog_bar=True, on_step=False, on_epoch=True)
#         print(f"[DEBUG] Discriminator Loss (avg): {avg_d_loss}")
#         print(f"[DEBUG] Generator Loss (avg): {avg_g_loss}")

#         return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn.utils import weight_norm, spectral_norm
from model.face_tts import FaceTTS
from model.feature_extractor import VoiceFeatureExtractor
from model.discriminator import SpectrogramDiscriminator

# ------------------------------------------
# FaceTTS with SpecDiffGAN Losses and Microbatching for Both Generator and Discriminator
# ------------------------------------------
class FaceTTSWithDiscriminator(FaceTTS):
    def __init__(self, _config):
        super().__init__(_config)
        self.config = _config
        self.discriminator = SpectrogramDiscriminator(_config)
        self.feature_extractor = VoiceFeatureExtractor(_config)
        
        # (2) Andere Loss-Funktion: statt MSELoss => BCEWithLogitsLoss
        # self.adv_criterion = nn.MSELoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        
        # Falls benötigt:
        # self.fm_criterion = nn.L1Loss()
        # self.recon_criterion = nn.L1Loss()
        
        # Automatische Optimierung deaktivieren, da wir manuell (und microbatch-weise) optimieren.
        self.automatic_optimization = False

    def compute_feature_matching_loss(self, real_features, fake_features):
        fm_loss = 0
        for real, fake in zip(real_features, fake_features):
            if real.size(-1) != fake.size(-1):
                if real.size(-1) > fake.size(-1):
                    real = real[..., :fake.size(-1)]
                else:
                    padding = real.size(-1) - fake.size(-1)
                    fake = F.pad(fake, (0, padding), mode="constant", value=0)
            fm_loss += self.fm_criterion(real, fake)
        return fm_loss

    def compute_pitch_loss(self, real_f0, fake_f0):
        if real_f0.size(-1) != fake_f0.size(-1):
            if real_f0.size(-1) > fake_f0.size(-1):
                real_f0 = real_f0[..., :fake_f0.size(-1)]
            else:
                padding = real_f0.size(-1) - fake_f0.size(-1)
                fake_f0 = F.pad(fake_f0, (0, padding), mode="constant", value=0)
        return self.recon_criterion(real_f0, fake_f0)

    def compute_energy_loss(self, real_energy, fake_energy):
        if real_energy.size(-1) != fake_energy.size(-1):
            if real_energy.size(-1) > fake_energy.size(-1):
                real_energy = real_energy[..., :fake_energy.size(-1)]
            else:
                padding = real_energy.size(-1) - fake_energy.size(-1)
                fake_energy = F.pad(fake_energy, (0, padding), mode="constant", value=0)
        return self.recon_criterion(real_energy, fake_energy)

    def configure_optimizers(self):
        # (3) Separates LR-Setting: disc_learning_rate statt (oder zusätzlich zu) learning_rate
        gen_lr = self.config["learning_rate"]
        disc_lr = self.config.get("disc_learning_rate", gen_lr)

        generator_optimizer = Adam(self.parameters(), lr=gen_lr)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=disc_lr)
        return [generator_optimizer, discriminator_optimizer]

    def training_step(self, batch, batch_idx):
        # Daten auf das Gerät verschieben
        x = batch['x'].to(self.device)
        x_len = batch['x_len']
        y = batch['y'].to(self.device)
        y_len = batch['y_len']
        spk = batch['spk'].to(self.device)
        B = x.shape[0]

        # ----- DISKRIMINATOR-UPDATE (MICROBATCHED) -----
        micro_batch_size = self.config.get("micro_batch_size", 16)
        n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size
        opt_disc = self.optimizers()[1]
        opt_disc.zero_grad()
        total_d_loss = 0.0

        for i in range(n_micro_batches):
            start = i * micro_batch_size
            end = min((i + 1) * micro_batch_size, B)
            x_mini = x[start:end]
            x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
            y_mini = y[start:end]
            y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
            spk_mini = spk[start:end]

            with amp.autocast():
                enc_out, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
                generated_mel_mini = dec_out[-1]
                real_logits, _ = self.discriminator(y_mini.unsqueeze(1))
                fake_logits, _ = self.discriminator(generated_mel_mini.detach().unsqueeze(1))
                # Falls der Discriminator mehrere Outputs liefert:
                real_logits = real_logits if isinstance(real_logits, torch.Tensor) else real_logits[0]
                fake_logits = fake_logits if isinstance(fake_logits, torch.Tensor) else fake_logits[0]
                loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
                loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                micro_loss_d = (loss_real + loss_fake) / 2

            # Optional: Skalierung, damit nach Akkumulation der Durchschnittswert entspricht
            micro_loss_d = micro_loss_d / n_micro_batches
            self.manual_backward(micro_loss_d)
            total_d_loss += micro_loss_d.item()

        opt_disc.step()
        avg_d_loss = total_d_loss  # Bereits als Durchschnitt berechnet

        # ----- GENERATOR-UPDATE (MICROBATCHED) -----
        micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
        n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen
        opt_gen = self.optimizers()[0]
        opt_gen.zero_grad()
        total_g_loss = 0.0

        for i in range(n_micro_batches_gen):
            start = i * micro_batch_size_gen
            end = min((i + 1) * micro_batch_size_gen, B)
            x_mini = x[start:end]
            x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
            y_mini = y[start:end]
            y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
            spk_mini = spk[start:end]

            with amp.autocast():
                enc_out, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
                generated_mel_mini = dec_out[-1]
                fake_logits, _ = self.discriminator(generated_mel_mini.unsqueeze(1))
                fake_logits = fake_logits if isinstance(fake_logits, torch.Tensor) else fake_logits[0]
                adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
                
                # Zusätzliche Diffusionsverluste
                dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini)
                
                # (1) Gewichtung des adversarial Loss via lambda_adv
                lambda_adv = self.config.get("lambda_adv", 1.0)
                micro_loss_g = lambda_adv * adv_loss + dur_loss + prior_loss + diff_loss + spk_loss

            micro_loss_g = micro_loss_g / n_micro_batches_gen
            self.manual_backward(micro_loss_g)
            total_g_loss += micro_loss_g.item()

        opt_gen.step()
        avg_g_loss = total_g_loss  # Durchschnittlicher Generator-Loss

        # Logge Schritt- und Epochenwerte separat:
        self.log("train/d_loss_step", avg_d_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/d_loss_epoch", avg_d_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/g_loss_step", avg_g_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/g_loss_epoch", avg_g_loss, prog_bar=True, on_step=False, on_epoch=True)
        print(f"[DEBUG] Discriminator Loss (avg): {avg_d_loss}")
        print(f"[DEBUG] Generator Loss (avg): {avg_g_loss}")

        return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import weight_norm, spectral_norm

# ------------------------------------------
# Spectrogram Discriminator with Aux Losses
# ------------------------------------------
class SpectrogramDiscriminator(pl.LightningModule):
    def __init__(self, _config):
        super(SpectrogramDiscriminator, self).__init__()
        self.config = _config
        self.LRELU_SLOPE = _config["lReLU_slope"]
        self.multi_speaker = _config["multi_spks"]
        self.residual_channels = _config["residual_channels"]
        self.use_spectral_norm = _config["use_spectral_norm"]
        norm_f = (weight_norm if self.use_spectral_norm else spectral_norm)

        self.conv_prev = norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            ]
        )

        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(norm_f(nn.Linear(self.residual_channels, 32)))

        self.conv_post = nn.ModuleList(
            [
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
                norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1))),
            ]
        )

    def forward(self, x, speaker_emb=None):
        print(f"[DEBUG] Initial x shape: {x.shape}")
        fmap = []

        x = self.conv_prev(x)
        print(f"[DEBUG] Shape after first Conv2D layer: {x.shape}")
        x = F.leaky_relu(x, self.LRELU_SLOPE)
        fmap.append(x)

        if self.multi_speaker and speaker_emb is not None:
            print(f"[DEBUG] Speaker embedding shape before processing: {speaker_emb.shape}")
            speaker_emb = self.spk_mlp(speaker_emb).unsqueeze(-1).expand(-1, -1, x.shape[-2]).unsqueeze(-1)
            print(f"[DEBUG] Speaker embedding shape after expansion: {speaker_emb.shape}")
            x = x + speaker_emb  # Inject speaker identity into the feature map

        for i, layer in enumerate(self.convs):
            x = layer(x)
            print(f"[DEBUG] Shape after Conv2D layer {i + 1}: {x.shape}")
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post[0](x)
        print(f"[DEBUG] Shape after post-processing Conv2D (1): {x.shape}")
        x = F.leaky_relu(x, self.LRELU_SLOPE)
        x = self.conv_post[1](x)
        print(f"[DEBUG] Shape after post-processing Conv2D (2): {x.shape}")
        x = torch.flatten(x, 1, -1)  # Flatten the final output
        print(f"[DEBUG] Final output shape of SpectrogramDiscriminator: {x.shape}")

        return fmap, x


import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import copy
from config import ex
from model.face_tts import FaceTTS
from data import _datamodules
from model.face_tts_w_discriminator import FaceTTSWithDiscriminator
import torch
import torch.distributed as dist

@ex.automain
def main(_config):
    print("[DEBUG] Starting training...")

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)
    print("[DEBUG] Data module initialized")

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
    model = FaceTTSWithDiscriminator(_config).to(torch.device("cuda"))

    print("[DEBUG] Model initialized")
   
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
    checkpoint = torch.load(_config["resume_from"], map_location="cuda")
    generator_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if "discriminator" not in k}
    model.load_state_dict(generator_state_dict, strict=False)

    print(f"Using {torch.cuda.device_count()} GPUs")

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        enable_model_summary=True,
        val_check_interval=_config["val_check_interval"],
    )
    print("[DEBUG] Trainer initialized")

    if not _config["test_only"]:
        print("[DEBUG] Starting training...")
        trainer.fit(model, datamodule=dm) 
    else:
        print("[DEBUG] Running test...")
        trainer.test(model, datamodule=dm)
