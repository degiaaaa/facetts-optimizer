import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn.utils import weight_norm, spectral_norm
from utils.scheduler import set_scheduler


from model.face_tts import FaceTTS
from model.feature_extractor import VoiceFeatureExtractor
from model.discriminator import SpectrogramDiscriminator

class FaceTTSWithDiscriminator(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.config = _config

        self.generator = FaceTTS(_config)
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
                signs = (2 * target - 1)  # {0->-1, 1->+1}
                return torch.mean(nn.ReLU()(1 - signs * logits))
            self.adv_criterion = hinge_loss_fn
        else:
            # Fallback:
            self.adv_criterion = nn.BCEWithLogitsLoss()

        # Disable automatic optimization (we'll update manually)
        self.automatic_optimization = False
        self.disc_scaler = torch.cuda.amp.GradScaler()

    def on_train_start(self):
        # Optionally freeze generator components at the start.
        if self.freeze_gen_epochs > 0:
            print(f"[INFO] Freezing generator for the first {self.freeze_gen_epochs} epochs.")
            for p in self.generator.encoder.parameters():
                p.requires_grad = False
            for p in self.generator.decoder.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        # Unfreeze generator after freeze_gen_epochs.
        if self.freeze_gen_epochs > 0 and self.current_epoch >= self.freeze_gen_epochs:
            print("[INFO] Unfreezing generator now.")
            for p in self.generator.encoder.parameters():
                p.requires_grad = True
            for p in self.generator.decoder.parameters():
                p.requires_grad = True
            self.freeze_gen_epochs = 0  # Do this only once


    #with Generator Scheduler
    def configure_optimizers(self):
        # Optionally use separate learning rates.
        gen_lr = self.config["learning_rate"]
        disc_lr = self.config.get("disc_learning_rate", gen_lr)

        gen_opt_list, gen_sched_list = set_scheduler(self.generator)
        generator_optimizer = gen_opt_list[0]
        generator_scheduler = gen_sched_list[0]

        disc_eps = self.config.get("disc_eps", 1e-8)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=disc_lr, betas=(self.config["disc_betas_0"], self.config["disc_betas_1"]), eps=disc_eps)
        # Return optimizers and scheduler configurations as lists.
        return [generator_optimizer, discriminator_optimizer], [generator_scheduler]

    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len, spk = (
            batch["x"],
            batch["x_len"],
            batch["y"],
            batch["y_len"],
            batch["spk"],
        )

        dur_loss, prior_loss, diff_loss, spk_loss = self.generator.compute_loss(
            x, x_len, y, y_len, spk=spk, out_size=self.config["out_size"]
        )

        val_loss = dur_loss + prior_loss + diff_loss + spk_loss

        self.log("val/duration_loss", dur_loss, prog_bar=True)
        self.log("val/prior_loss", prior_loss, prog_bar=True)
        self.log("val/diffusion_loss", diff_loss, prog_bar=True)
        self.log("val/spk_loss", spk_loss, prog_bar=True)
        self.log("val/total_loss", val_loss, prog_bar=True)

        return val_loss
        
    def training_step(self, batch, batch_idx):
        x = batch['x'].to(self.device)
        x_len = batch['x_len']
        y = batch['y'].to(self.device)
        y_len = batch['y_len']
        spk = batch['spk'].to(self.device)
        B = x.shape[0]

        micro_batch_size = self.config.get("micro_batch_size", 16)
        n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size

        opt_gen, opt_disc = self.optimizers()
        train_disc = self.current_epoch >= self.warmup_disc_epochs

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
                    with torch.no_grad():
                        _, dec_out, _ = self.generator.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
                    fake_mel = dec_out[-1]

                    _, real_logits = self.discriminator(y_mini.unsqueeze(1))
                    _, fake_logits = self.discriminator(fake_mel.detach().unsqueeze(1))
                    loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
                    loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (loss_real + loss_fake)

                self.disc_scaler.scale(d_loss / n_micro_batches).backward()
            self.disc_scaler.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.disc_scaler.step(opt_disc)
            self.disc_scaler.update()
            avg_d_loss = d_loss.item()
        else:
            avg_d_loss = 0.0

        micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
        n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen

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

            enc_out, dec_out, _ = self.generator.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
            fake_mel = dec_out[-1]
            if train_disc:
                _, fake_logits = self.discriminator(fake_mel.unsqueeze(1))
                adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
            else:
                adv_loss = torch.tensor(0.0, device=self.device)

            dur_loss, prior_loss, diff_loss, spk_loss = self.generator.compute_loss(
                x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini
            )

            g_loss = (self.lambda_adv * adv_loss) + dur_loss + prior_loss + diff_loss + spk_loss

            self.manual_backward(g_loss / n_micro_batches_gen)
            total_g_loss += g_loss.item()

        torch.nn.utils.clip_grad_norm_(list(self.generator.encoder.parameters()) + list(self.generator.decoder.parameters()), max_norm=1.0)
        opt_gen.step()
        avg_g_loss = total_g_loss

        self.log("train/d_loss_step", avg_d_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/g_loss_step", avg_g_loss, prog_bar=True, on_step=True, on_epoch=False)

        if (batch_idx + 1) == len(self.trainer.datamodule.train_dataloader()):
            self.log("train/d_loss_epoch", avg_d_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/g_loss_epoch", avg_g_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}