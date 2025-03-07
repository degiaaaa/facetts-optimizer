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
        self.disc_loss_type = _config["disc_loss_type"]

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

        # Debugging Statements
        print(f"[DEBUG] Batch {batch_idx} - x.shape: {x.shape}, y.shape: {y.shape}, spk.shape: {spk.shape}")
        print(f"[DEBUG] x_len: {x_len}, y_len: {y_len}")

        # Check für ungültige Werte
        if torch.any(y_len < 0) or torch.any(torch.isnan(y_len)):
            print(f"[ERROR] contains nan-values: {y_len}")
            raise ValueError("y_lengths contains nan-values!")

        # Get microbatch sizes (defined unconditionally)
        micro_batch_size = self.config.get("micro_batch_size", 16)
        micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
        n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size
        n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen

        # Get optimizers.
        opt_gen, opt_disc = self.optimizers()

        # DEBUG: Prüfe, ob Discriminator nach Warmup aktiv wird
        if self.current_epoch == self.warmup_disc_epochs:
            print(f"[DEBUG] Discriminator will be activated! Epoch: {self.current_epoch}")

        # Determine whether to update the discriminator based on a warm-up.
        train_disc = self.current_epoch >= self.warmup_disc_epochs

        ### Discriminator Update ###
        if train_disc:
            #zero out the gradients to prevent mixing gradients from previous iterations
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

                    print(f"[DEBUG] Fake Mel shape: {fake_mel.shape}")
                    # Run discriminator on real and fake.
                    _, real_logits = self.discriminator(y_mini.unsqueeze(1))
                    _, fake_logits = self.discriminator(fake_mel.detach().unsqueeze(1)) #fake not recorded to gradient computation (detach)
                    loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
                    loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (loss_real + loss_fake)
                    # # 🛠 Histogram-Logging für den Discriminator
                    # self.logger.experiment.add_histogram("Discriminator/Real_Logits", real_logits.detach().cpu(), self.global_step)
                    # self.logger.experiment.add_histogram("Discriminator/Fake_Logits", fake_logits.detach().cpu(), self.global_step)
                self.manual_backward(d_loss / n_micro_batches)  # Accumulate gradients
                total_d_loss += d_loss.item()
            opt_disc.step()
            avg_d_loss = total_d_loss
            print(f"[DEBUG] Batch {batch_idx} -> Discriminator Loss: {avg_d_loss}")
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

                # # 🛠 Logging für adversarialen Verlust und lambda_adv
                # self.log("train/adv_loss", adv_loss.item(), prog_bar=True, on_step=True, on_epoch=False)
                # self.log("train/lambda_adv", self.lambda_adv, prog_bar=True, on_step=False, on_epoch=True)

                # Compute the original FaceTTS losses.
                dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(
                    x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini
                )
                

                # Combine all losses.
                g_loss = (self.lambda_adv * adv_loss) + dur_loss + prior_loss + diff_loss  + (spk_loss)
            self.manual_backward(g_loss / n_micro_batches_gen)  # Accumulate gradients
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

        #return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}
        # # --- Monitoring: Zusätzliche Debug-/Monitoring-Ausgaben ---
        # # Hier loggen wir y_lengths, y_max_length und y_max_length_.
        # with torch.no_grad():
        #     # Für Monitoring verwenden wir den ersten Sample der Charge:
        #     _, _, attn = self.forward(x[:1], x_len[:1], self.config['timesteps'], spk=spk[:1])
        #     # Erneute Berechnung (wie in forward) für y_lengths, y_max_length und y_max_length_:
        #     mu_x, logw, x_mask = self.encoder(x[:1], x_len[:1], spk[:1])
        #     w = torch.exp(logw) * x_mask
        #     w_ceil = torch.ceil(w) * self.config.get("length_scale", 1.0)
        #     y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        #     y_max_length = int(y_lengths.max())
        #     from model.utils import fix_len_compatibility
        #     y_max_length_ = fix_len_compatibility(y_max_length)
        #     global_step = self.global_step if hasattr(self, "global_step") else 0
        #     self.logger.experiment.add_text("Monitoring/y_lengths", str(y_lengths.tolist()), global_step=global_step)
        #     self.logger.experiment.add_text("Monitoring/y_max_length", str(y_max_length), global_step=global_step)
        #     self.logger.experiment.add_text("Monitoring/y_max_length_", str(y_max_length_), global_step=global_step)

        #     # Plot und logge ein Mel‑Spektrogramm des Fake-Mels:
        #     try:
        #         mel_plot = plot_tensor(fake_mel[0].detach().cpu())
        #         self.logger.experiment.add_image("FakeMelSpec", mel_plot, global_step=global_step, dataformats='HWC')
        #     except Exception as e:
        #         print("[MONITOR] Could not plot fake mel spectrogram:", e)

        return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}