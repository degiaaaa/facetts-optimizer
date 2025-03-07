import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import copy
import torch

from config import ex
from data import _datamodules  # Hier wird dein LRS3DataModule importiert
from model.face_tts import FaceTTS
from model.face_tts_w_discriminator import FaceTTSWithDiscriminator

@ex.automain
def main(_config):
    """
    Unified script that trains either FaceTTS or FaceTTSWithDiscriminator 
    based on _config["use_gan"].
    """
    print("[DEBUG] Starting script...")

    # Copy the config so we don't mutate it globally
    _config = copy.deepcopy(_config)

    # Set random seed
    pl.seed_everything(_config["seed"])

    # --------------------------------------------------------------------------
    # Data Module
    # --------------------------------------------------------------------------
    dm = _datamodules["dataset_" + _config["dataset"]](_config)
    # Stelle sicher, dass das DataModule eingerichtet wird (sodass test_dataset gesetzt wird)
    dm.setup("test")
    print("[DEBUG] Data module initialized")

    # --------------------------------------------------------------------------
    # Callbacks & Logging
    # --------------------------------------------------------------------------
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
    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)
    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]

    # --------------------------------------------------------------------------
    # Model selection based on _config["use_gan"]
    # --------------------------------------------------------------------------
    use_gan = bool(_config["use_gan"])  # 0 => False, 1 => True
    if use_gan:
        print("[INFO] use_gan=True -> using FaceTTSWithDiscriminator")
        model = FaceTTSWithDiscriminator(_config).to(torch.device("cuda"))
    else:
        print("[INFO] use_gan=False -> using FaceTTS")
        model = FaceTTS(_config).to(torch.device("cuda"))

    print("[DEBUG] Model initialized")

    # --------------------------------------------------------------------------
    # Testlauf mit definiertem Input vor Checkpoint-Laden (Batch-Größe reduziert)
    # --------------------------------------------------------------------------
    model.eval()  # Sicherstellen, dass das Modell im Evaluierungsmodus ist
    try:
        # Nutze den test_dataloader (jetzt sollte test_dataset vorhanden sein)
        test_loader = dm.test_dataloader() if hasattr(dm, "test_dataloader") else dm.train_dataloader()
        test_batch = next(iter(test_loader))
        # Reduziere die Batch-Größe: verwende nur den ersten Sample aus dem Batch
        x_test = test_batch["x"][0:1].to(torch.device("cuda"))
        x_len_test = test_batch["x_len"][0:1]
        spk_test = test_batch["spk"][0:1].to(torch.device("cuda"))
        n_timesteps = _config.get("timesteps", 10)
        
        with torch.no_grad():
            ref_encoder_out, ref_decoder_out, ref_attn = model(x_test, x_len_test, n_timesteps, spk=spk_test)
        ref_mean = ref_encoder_out.mean().item()
        print("[DEBUG] Referenz: Encoder output mean (vor Checkpoint):", ref_mean)
    except Exception as e:
        print("[WARNING] Testlauf vor Checkpoint-Laden konnte nicht durchgeführt werden:", e)

    # --------------------------------------------------------------------------
    # GPU / Trainer settings
    # --------------------------------------------------------------------------
    if isinstance(_config["num_gpus"], int):
        num_gpus = _config["num_gpus"]
    else:
        num_gpus = len(_config["num_gpus"])

    grad_steps = _config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # --------------------------------------------------------------------------
    # Loading checkpoint:
    # a) If GAN is used, load only generator parts
    # b) If not using GAN, load the entire state dict
    # --------------------------------------------------------------------------
    print(f"[DEBUG] Loading checkpoint from {_config['resume_from']}")
    checkpoint = torch.load(_config["resume_from"], map_location="cuda")

    if use_gan:
        # Filter out discriminator keys und lade nur die Generator-Gewichte
        generator_state_dict = {
            k: v for k, v in checkpoint['state_dict'].items() if "discriminator" not in k
        }
        model.load_state_dict(generator_state_dict, strict=False)
        print("[DEBUG] Loaded generator weights (discriminator keys ignored).")
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("[DEBUG] Loaded entire model state_dict.")

    # --------------------------------------------------------------------------
    # Testlauf nach dem Laden des Checkpoints (Batch-Größe reduziert)
    # --------------------------------------------------------------------------
    model.eval()  # Nochmals sicherstellen, dass das Modell im Evaluierungsmodus ist
    try:
        with torch.no_grad():
            ckpt_encoder_out, ckpt_decoder_out, ckpt_attn = model(x_test, x_len_test, n_timesteps, spk=spk_test)
        ckpt_mean = ckpt_encoder_out.mean().item()
        print("[DEBUG] Checkpoint: Encoder output mean (nach Checkpoint):", ckpt_mean)
        diff = abs(ckpt_mean - ref_mean)
        print("[DEBUG] Absolute difference between pre- and post-checkpoint encoder output means:", diff)
        # Prüfe anhand eines Toleranzwerts, ob die Ergebnisse konsistent sind
        if diff < 1e-6:
            print("[INFO] Die Ergebnisse sind konsistent – Checkpoint-Gewichte wurden nich übernommen.")
        else:
            print("[WARNING] Die Ergebnisse weichen signifikant ab. Checkpoint-Gewichte werden geladen.")
    except Exception as e:
        print("[WARNING] Testlauf nach Checkpoint-Laden konnte nicht durchgeführt werden:", e)

    print(f"[INFO] Using {torch.cuda.device_count()} GPU(s)")

    # --------------------------------------------------------------------------
    # Trainer setup
    # --------------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        enable_model_summary=True,
        val_check_interval=_config["val_check_interval"],
    )
    print("[DEBUG] Trainer initialized")

    # --------------------------------------------------------------------------
    # Train or Test
    # --------------------------------------------------------------------------
    if not _config["test_only"]:
        print("[DEBUG] Starting training...")
        trainer.fit(model, datamodule=dm)
    else:
        print("[DEBUG] Running test...")
        trainer.test(model, datamodule=dm)
