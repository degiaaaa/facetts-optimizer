import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import copy
import cv2
import numpy as np
from tqdm import tqdm
from config import ex
from model.face_tts import FaceTTS
from data.lrs3_dataset import LRS3Dataset
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils.tts_util import intersperse
from scipy.io.wavfile import write

# Set test dataset directory
LRS2_TEST_DIR = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/test"

@ex.automain
def main(_config):
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    
    print("######## Initializing TTS model")
    model = FaceTTS(_config).cuda()
    
    # Read `use_gan` from config without overwriting (0 = False, 1 = True)
    use_gan = _config.get("use_gan")

    # Select correct checkpoint based on `use_gan`
    checkpoint_path = _config.get("infr_resume_from_gan") if use_gan else _config.get("infr_resume_from_orig")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"######## Loading checkpoint from {checkpoint_path}")

    # Load full checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Check if the checkpoint contains GAN-specific discriminator weights
    has_discriminator = any(k.startswith("discriminator") for k in checkpoint['state_dict'].keys())

    if has_discriminator:
        print("######## GAN checkpoint detected. Loading only generator weights.")
        # Remove discriminator weights before loading
        generator_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith("discriminator")}
        model.load_state_dict(generator_state_dict, strict=False)
    else:
        print("######## Standard FaceTTS checkpoint detected.")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.zero_grad()

    print("######## Initializing HiFi-GAN")
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().cuda()

    cmu = cmudict.CMUDict(_config['cmudict_path'])

    # Select correct output directory based on `use_gan`
    output_dir = _config.get("output_dir_gan") if use_gan else _config.get("output_dir_orig")

    # Decide whether to use a custom face image or extract from dataset
    if _config['use_custom']:      
        print(f"######## Load custom face image: {_config['test_faceimg']}")
        spk = cv2.imread(_config['test_faceimg'])
        spk = cv2.resize(spk, (224, 224))
        spk = np.transpose(spk, (2, 0, 1))
        spk = torch.FloatTensor(spk).unsqueeze(0).to(model.device)
    else:
        print(f"######## Load speaker from dataset: {_config['dataset']}")
        dataset = LRS3Dataset(split="test", config=_config)
        
        # Load a face image from dataset using `load_random_frame`
        sample_file = os.listdir(LRS2_TEST_DIR)[0]  # Select a random speaker
        sample_path = os.path.join(LRS2_TEST_DIR, sample_file)
        
        spk = dataset.load_random_frame(LRS2_TEST_DIR, sample_file)
        if spk is None:
            raise ValueError(f"Failed to load speaker face frame from {sample_path}")
        
        spk = torch.FloatTensor(spk[0]).unsqueeze(0).to(model.device)

    # Process dataset
    with torch.no_grad():
        for speaker in os.listdir(LRS2_TEST_DIR):
            speaker_dir = os.path.join(LRS2_TEST_DIR, speaker)

            for filename in os.listdir(speaker_dir):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(speaker_dir, filename)
                    text_path = video_path.replace(".mp4", ".txt")

                    # Load text description
                    if not os.path.exists(text_path):
                        print(f"Warning: Missing transcript for {video_path}")
                        continue

                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.readline().strip()

                    print(f"Processing {video_path}")

                    # Convert text to sequence
                    x = torch.LongTensor(
                        intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
                    ).to(model.device)[None]

                    x_len = torch.LongTensor([x.size(-1)]).to(model.device)
                    y_enc, y_dec, attn = model.forward(
                        x,
                        x_len,
                        n_timesteps=_config["timesteps"],
                        temperature=1.5,
                        stoc=False,
                        spk=spk,
                        length_scale=0.91,
                    )

                    audio = (
                        vocoder.forward(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy()
                        * 32768
                    ).astype(np.int16)
                    
                    output_speaker_dir = os.path.join(output_dir, speaker)
                    os.makedirs(output_speaker_dir, exist_ok=True)

                    output_path = os.path.join(output_speaker_dir, f"{filename.replace('.mp4', '.wav')}")
                    write(output_path, _config["sample_rate"], audio)

    print(f"######## Done inference. Check '{output_dir}' folder")
