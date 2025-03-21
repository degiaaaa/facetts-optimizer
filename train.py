import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import copy
import torch

from config import ex
from data import _datamodules
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
    print("[DEBUG] Data module initialized")

    # --------------------------------------------------------------------------
    # Checkpoint Directory and Prevention of Overwriting resume_from
    # --------------------------------------------------------------------------
    # checkpoint_dir = _config.get("checkpoint_dir", "/mnt/qb/work/butz/bst080/faceGANtts/checkpoints")
    # if _config["resume_from"] and os.path.dirname(_config["resume_from"]) == checkpoint_dir:
    #     raise ValueError(f"ERROR: resume_from ({_config['resume_from']}) is inside checkpoint_dir ({checkpoint_dir}). "
    #                      "This could lead to overwriting! Choose a different checkpoint directory.")

    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        # dirpath=checkpoint_dir,  # Separate directory for new checkpoints
        # filename="epoch={epoch}-step={step}",
        save_weights_only=False,
        save_top_k=3,  # Keeps last 3 best checkpoints
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
        every_n_epochs=1  # Save at every epoch
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
    # GPU / Trainer settings
    # --------------------------------------------------------------------------
    num_gpus = _config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"])
    grad_steps = _config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # --------------------------------------------------------------------------
    # Loading checkpoint - Prevent Overwriting
    # --------------------------------------------------------------------------
    if os.path.exists(_config["resume_from"]):
        print(f"[INFO] Loading checkpoint from {_config['resume_from']}")
        checkpoint = torch.load(_config["resume_from"], map_location="cuda")
        checkpoint.pop('callbacks', None)  # Remove callbacks to prevent changes

        if use_gan:
            generator_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if "discriminator" not in k}
            model.load_state_dict(generator_state_dict, strict=False)
            print("[INFO] Loaded generator weights (discriminator keys ignored).")
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("[INFO] Loaded entire model state_dict.")
    else:
        print(f"[WARNING] No checkpoint found at {_config['resume_from']}. Training from scratch.")

    #print("[DEBUG] Starting training without checkpoint...")

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
    trainer.logger.log_hyperparams(_config)
    
    # --------------------------------------------------------------------------
    # Train or Test
    # --------------------------------------------------------------------------
    if not _config["test_only"]:
        print("[DEBUG] Starting training...")
        trainer.fit(model, datamodule=dm)
    else:
        print("[DEBUG] Running test...")
        trainer.test(model, datamodule=dm)
