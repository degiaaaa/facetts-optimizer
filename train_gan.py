
import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
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

    # if dist.is_available() and dist.is_initialized():
    #     print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")

    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    # print("Memory cleared before training starts!")
    # torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()
    # # Limit each GPU to 80% memory usage
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i} - {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB allocated, {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB reserved")

    # for i in range(torch.cuda.device_count()):
    #     torch.cuda.set_per_process_memory_fraction(0.8, device=i)
    # print("✅ GPU memory fraction set to 80% per device")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"Total GPUs: {torch.cuda.device_count()}")
    # print(f"Active GPU: {torch.cuda.current_device()}")

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)
    print("[DEBUG] Data module initialized")

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
        #gpus=_config["num_gpus"],
        accelerator="gpu", 
        devices= _config["num_gpus"],#_config["num_gpus"], #torch.cuda.device_count(), 
        num_nodes=_config["num_nodes"],
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True), #FACETTS original find_unused_parameters=True
        #precision=16, 
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps, #grad_steps test max(1, _config["batch_size"] // (_config["per_gpu_batchsize"] * torch.cuda.device_count() * _config["num_nodes"])),
        log_every_n_steps=50,
        #flush_logs_every_n_steps=50,
        #weights_summary="top",
        enable_model_summary=True,
        val_check_interval=_config["val_check_interval"],
    )
    print("[DEBUG] Trainer initialized")

    if not _config["test_only"]:
        print("[DEBUG] Starting training...")
        trainer.fit(model, datamodule=dm) #, ckpt_path=_config["resume_from"]
    else:
        print("[DEBUG] Running test...")
        trainer.test(model, datamodule=dm) #, ckpt_path=_config["resume_from"]