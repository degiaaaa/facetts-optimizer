
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.optim import Adam
from model.face_tts import FaceTTS
#from model.syncnet_hifigan import SyncNet
#from model.diffusion import Diffusion
#from model.utils import sequence_mask

# Define SpectrogramDiscriminator
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
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)

# Adjusting training loop for debugging and avoiding GPU OOM issues
class FaceTTSWithDiscriminator(FaceTTS):
    def __init__(self, config):
        super().__init__(config)
        self.discriminator = SpectrogramDiscriminator()
        self.adv_criterion = nn.MSELoss()

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

        # Free memory from intermediate tensors
        torch.cuda.empty_cache()

        # Discriminator step
        if optimizer_idx == 1:
            real_logits = self.discriminator(y.unsqueeze(1))
            fake_logits = self.discriminator(generated_mel.detach().unsqueeze(1))

            real_loss = self.adv_criterion(real_logits, torch.ones_like(real_logits))
            fake_loss = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
            return d_loss

        # Generator step
        elif optimizer_idx == 0:
            fake_logits = self.discriminator(generated_mel.unsqueeze(1))
            adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))

            # Combine with existing losses
            dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(x, x_len, y, y_len, spk=spk)

            g_loss = adv_loss + dur_loss + prior_loss + diff_loss + spk_loss
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

    os.makedirs(_config["local_checkpoint_dir"], exist_ok=True)

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
        devices=_config["num_gpus"],
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
