import torch
import torchaudio
import os
import random
from model.utils import fix_len_compatibility
from utils.tts_util import intersperse, parse_filelist
from text import cmudict
from utils.mel_spectrogram import mel_spectrogram
import cv2
import numpy as np

from text import text_to_sequence
from text import symbols

#NEW for LRS2 and ML-Cloud Cluster
import time
import noisereduce as nr


class LRS3Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "", config=None):
        assert split in ["train", "val", "test"]
        super().__init__()

        self.split = split
        self.config = config

        self.cmudict = cmudict.CMUDict(self.config["cmudict_path"])

        if self.split == "train":
            self.filelist = self.config["lrs3_train"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "trainval")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/trainval")
        elif self.split == "val":
            self.filelist = self.config["lrs3_val"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "trainval")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/trainval")
        elif self.split == "test":
            self.filelist = self.config["lrs3_test"]
            self.video_dir = os.path.join(self.config["lrs3_path"], "test")
            self.audio_dir = os.path.join(self.config["lrs3_path"], "wav/test")

        # Load datalist
        with open(self.filelist) as listfile:
            self.data_list = listfile.readlines()

        print(f"{split} set: ", len(self.data_list))

        spk_list = [data.split("\n")[0].split("/")[0] for data in self.data_list]
        spk_list = set(spk_list)
        print(f"{len(spk_list)=}")
        
        self.spk_list = dict()
        for i, spk in enumerate(spk_list):
            self.spk_list[spk] = i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        name = self.data_list[index].split("\n")[0]
        vidname = name + ".mp4"
        textpath = name + ".txt"
        print(f"Loading audio from: {os.path.join(self.audio_dir, name + '.wav')}")

        aud, sr = torchaudio.load(os.path.join(self.audio_dir, name + ".wav"))
        
        assert (sr == self.config["sample_rate"]), "sampling rate should be 16k."
        
        #For LRS2 Data apply denoising
        aud_np = aud.numpy()
        aud_denoised = nr.reduce_noise(y=aud_np, sr=sr, prop_decrease=self.config["denoise_factor"])
        aud = torch.tensor(aud_denoised)

        aud = mel_spectrogram(
            aud,
            self.config["n_fft"],
            self.config["n_mels"],
            self.config["sample_rate"],
            self.config["hop_len"],
            self.config["win_len"],
            self.config["f_min"],
            self.config["f_max"],
            center=False,
        )
        
        text = (
            open(os.path.join(self.video_dir, textpath))
            #.readlines()[0] #NEW because of text transformation
            #.split(":")[1] #NEW because of text transformation
            .read() #NEW because of text transformation
            .strip()
        )
        
        if isinstance(text, type(None)):
            print(text)
            print(name)
        else:
            text += "."

        img = self.load_random_frame(self.video_dir, f"{name}.mp4", 1)
        txt = self.loadtext(text, self.cmudict, self.config["add_blank"])
        spk = self.spk_list[name.split("/")[0]]

        return {
            "spk_id": torch.LongTensor([int(spk)]),
            "spk": img,
            "y": aud.squeeze(),
            "x": txt,
            "name": name,
        }

    def loadtext(self, text, cmudict, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=cmudict)
        if add_blank:
            text_norm = intersperse(text_norm, len(symbols))
        text_norm = torch.IntTensor(text_norm)
        return text_norm


    # def load_random_frame(self, datadir, filename, len_frame=1):
    #     # len_frame == -1: load all frames
    #     # else: load random index frame with len_frames
    #     cap = cv2.VideoCapture(os.path.join(datadir, filename))
        
    #     # Check if the video opened successfully
    #     if not cap.isOpened():
    #         raise FileNotFoundError(f"Unable to open video file: {os.path.join(datadir, filename)}")

    #     nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #     if len_frame == -1:
    #         ridx = 0
    #         loadframe = nframes
    #     else:
    #         # Ensure ridx is within bounds
    #         if nframes <= 2:
    #             raise ValueError(f"Video file {filename} does not have enough frames.")
    #         ridx = random.randint(2, nframes - len_frame)
    #         loadframe = len_frame
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, ridx)

    #     imgs = []
    #     target_size = (224, 224)  # Resizing target size

    #     for i in range(loadframe):
    #         ret, img = cap.read()
            
    #         if not ret:
    #             raise ValueError(f"Failed to read frame {ridx + i} from video {filename}")

    #         # Handle grayscale or single-channel images
    #         if len(img.shape) == 2:  # Grayscale image
    #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         elif img.shape[2] == 1:  # Single-channel image
    #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    #         img = cv2.resize(img, target_size)
    #         imgs.append(img)

    #     cap.release()

    #     # Stack frames and change the shape to match expected output
    #     imgs = np.stack(imgs, axis=0)  # Stack frames along the first axis
    #     imgs = np.transpose(imgs, (0, 3, 1, 2))  # Convert to (num_frames, channels, height, width)

    #     return imgs

    def load_random_frame(self, datadir, filename, len_frame=1):
        """
        Load a random frame from a video file, with retry mechanism for handling file access issues.
        """
        max_attempts=5
        attempt = 0
        while attempt < max_attempts:
            try:
                cap = cv2.VideoCapture(os.path.join(datadir, filename))

                if not cap.isOpened():
                    raise FileNotFoundError(f"Unable to open video file: {os.path.join(datadir, filename)}")

                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if len_frame == -1:
                    ridx = 0
                    loadframe = nframes
                else:
                    if nframes <= 2:
                        raise ValueError(f"Video file {filename} does not have enough frames.")
                    ridx = random.randint(2, nframes - len_frame)
                    loadframe = len_frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, ridx)

                imgs = []
                target_size = (224, 224)

                for i in range(loadframe):
                    ret, img = cap.read()
                    if not ret:
                        raise ValueError(f"Failed to read frame {ridx + i} from video {filename}")

                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    img = cv2.resize(img, target_size)
                    imgs.append(img)

                cap.release()

                imgs = np.stack(imgs, axis=0)  # Stack frames along the first axis
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # Convert to (num_frames, channels, height, width)
                return imgs

            except (FileNotFoundError, ValueError) as e:
                print(f"Attempt {attempt + 1} failed for {filename}: {e}")
                attempt += 1
                if attempt < max_attempts:
                    print("Retrying...")
                    time.sleep(10)
                else:
                    raise FileNotFoundError(f"Failed to load {filename} after {max_attempts} attempts.")

        return None  # Fallback (unreachable due to raise above)



class TextMelVideoBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item["y"], item["x"], item["spk"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        # Convert all numpy arrays to torch tensors
        for i in range(len(spk)):
            if isinstance(spk[i], np.ndarray):
                spk[i] = torch.tensor(spk[i], dtype=torch.float32)
            # Check if all elements in spk are torch tensors
            elif not isinstance(spk[i], torch.Tensor):
                raise TypeError(f"Unexpected type for spk[{i}]: {type(spk[i])}")

        spk = torch.cat(spk, dim=0)
        return {
            "x": x,
            "x_len": x_lengths,
            "y": y,
            "y_len": y_lengths,
            "spk": spk,
        }
