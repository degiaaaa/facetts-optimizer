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
        self.LRELU_SLOPE = _config["disc_lrelu_slope"]
        self.multi_speaker = _config["multi_spks"]
        self.residual_channels = _config["residual_channels"]
        self.use_spectral_norm = _config["use_spectral_norm"]
        self.base_channels = _config["disc_base_channels"]
        self.num_layers = _config["disc_num_layers"]

        norm_f = (weight_norm if self.use_spectral_norm else spectral_norm)

        self.conv_prev = norm_f(nn.Conv2d(1, self.base_channels, (3, 9), padding=(1, 4)))


        # self.convs = nn.ModuleList(
        #     [
        #         norm_f(nn.Conv2d(self.base_channels, self.base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
        #         norm_f(nn.Conv2d(self.base_channels, self.base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
        #         norm_f(nn.Conv2d(self.base_channels, self.base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
        #     ]
        # )
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                norm_f(nn.Conv2d(self.base_channels, self.base_channels, (3, 9), stride=(1, 2), padding=(1, 4)))
                )

        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(norm_f(nn.Linear(self.residual_channels, 32)))

        self.conv_post = nn.ModuleList(
            [
                norm_f(nn.Conv2d(self.base_channels, self.base_channels, (3, 3), padding=(1, 1))),
                norm_f(nn.Conv2d(self.base_channels, 1, (3, 3), padding=(1, 1))),
            ]
        )

    def forward(self, x, speaker_emb=None):
        print(f"[DEBUG] Initial x shape: {x.shape}")
        fmap = []

       # x = x.unsqueeze(1)  # Add channel dimension
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

