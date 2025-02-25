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

class FaceTTSWithDiscriminator(FaceTTS):
    def __init__(self, _config):
        super().__init__(_config)
        self.config = _config

        # Instantiate discriminator and feature extractor.
        self.discriminator = SpectrogramDiscriminator(_config)
        #self.feature_extractor = VoiceFeatureExtractor(_config)

        # Hyperparameters for adversarial training.
        self.lambda_adv = _config["lambda_adv"]  # small adversarial weight initially
        self.warmup_disc_epochs = _config["warmup_disc_epochs"]  # skip disc updates for these epochs
        self.freeze_gen_epochs = _config["freeze_gen_epochs"] # freeze generator for these epochs
        #self.adv_criterion = nn.BCEWithLogitsLoss()
        self.disc_loss_type = _config["disc_loss_type"]
        #self.speaker_loss_weight = _config["speaker_loss_weight"]

        # Loss-Funktion ggf. abhängig von disc_loss_type:
        if self.disc_loss_type == "bce":
            self.adv_criterion = nn.BCEWithLogitsLoss()
        elif self.disc_loss_type == "mse":
            self.adv_criterion = nn.MSELoss()
        elif self.disc_loss_type == "hinge":
            # Beispielhaft, Hinge-Loss kann man so implementieren:
            def hinge_loss_fn(logits, target):
                # target ist hier 1 oder 0, also +1 / -1 interpretieren
                # Simple Implementation ...
                signs = (2 * target - 1)  # {0->-1, 1->+1}
                return torch.mean(nn.ReLU()(1 - signs * logits))
            self.adv_criterion = hinge_loss_fn
        else:
            # Fallback:
            self.adv_criterion = nn.BCEWithLogitsLoss()

        # Disable automatic optimization (we'll update manually)
        self.automatic_optimization = False

    def on_train_start(self):
        # Optionally freeze generator components at the start.
        if self.freeze_gen_epochs > 0:
            print(f"[INFO] Freezing generator for the first {self.freeze_gen_epochs} epochs.")
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        # Unfreeze generator after freeze_gen_epochs.
        if self.freeze_gen_epochs > 0 and self.current_epoch >= self.freeze_gen_epochs:
            print("[INFO] Unfreezing generator now.")
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True
            self.freeze_gen_epochs = 0  # Do this only once

    def configure_optimizers(self):
        # Optionally use separate learning rates.
        gen_lr = self.config["learning_rate"]
        disc_lr = self.config.get("disc_learning_rate", gen_lr)
        generator_optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=gen_lr
        )
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=disc_lr)
        return [generator_optimizer, discriminator_optimizer]

    def training_step(self, batch, batch_idx):
        # Move data to device.
        x = batch['x'].to(self.device)
        x_len = batch['x_len']
        y = batch['y'].to(self.device)
        y_len = batch['y_len']
        spk = batch['spk'].to(self.device)
        B = x.shape[0]

        # Get microbatch sizes (defined unconditionally)
        micro_batch_size = self.config.get("micro_batch_size", 16)
        micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
        n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size
        n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen

        # Get optimizers.
        opt_gen, opt_disc = self.optimizers()

        # Determine whether to update the discriminator based on a warm-up.
        train_disc = self.current_epoch >= self.warmup_disc_epochs

        ### Discriminator Update ###
        if train_disc:
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
                    # Generate fake mel-spectrograms (without gradient update)
                    with torch.no_grad():
                        _, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
                    fake_mel = dec_out[-1]
                    # Run discriminator on real and fake.
                    _, real_logits = self.discriminator(y_mini.unsqueeze(1))
                    _, fake_logits = self.discriminator(fake_mel.detach().unsqueeze(1))
                    loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
                    loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (loss_real + loss_fake)
                self.manual_backward(d_loss / n_micro_batches)
                total_d_loss += d_loss.item()
            opt_disc.step()
            avg_d_loss = total_d_loss
        else:
            avg_d_loss = 0.0

        ### Generator Update ###
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
                fake_mel = dec_out[-1]
                # If discriminator is active, compute adversarial loss.
                if train_disc:
                    _, fake_logits = self.discriminator(fake_mel.unsqueeze(1))
                    adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
                else:
                    adv_loss = torch.tensor(0.0, device=self.device)

                # Compute the original FaceTTS losses.
                dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(
                    x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini
                )
                #spk_loss_weighted = self.speaker_loss_weight * spk_loss

                # Combine all losses.
                g_loss = (self.lambda_adv * adv_loss) + dur_loss + prior_loss + diff_loss  + spk_loss #spk_loss_weighted
            self.manual_backward(g_loss / n_micro_batches_gen)
            total_g_loss += g_loss.item()
        opt_gen.step()
        avg_g_loss = total_g_loss

        # Log step and epoch metrics.
        self.log("train/d_loss_step", avg_d_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/g_loss_step", avg_g_loss, prog_bar=True, on_step=True, on_epoch=False)
        if (batch_idx + 1) == len(self.trainer.datamodule.train_dataloader()):
            self.log("train/d_loss_epoch", avg_d_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/g_loss_epoch", avg_g_loss, prog_bar=True, on_step=False, on_epoch=True)

        print(f"[DEBUG] Discriminator Loss (avg): {avg_d_loss}")
        print(f"[DEBUG] Generator Loss (avg): {avg_g_loss}")

        return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}
