import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import re
import transforms as transforms

class CupDataset(Dataset):
    def __init__(self, split, arg_obj):
        self.fps = arg_obj.fps
        self.aug = arg_obj.augmentation.lower()
        self.split = split
        self.channels = arg_obj.channels
        self.frames_per_clip = int(arg_obj.fpc)
        self.speed_slow = arg_obj.speed_slow
        self.speed_fast = arg_obj.speed_fast
        self.data_path = arg_obj.train_path if split == 'train' else arg_obj.val_path
        self.type = arg_obj.data_type
        self.condition = arg_obj.condition
        self.cam = arg_obj.cam
        self.wavelength = arg_obj.wavelength

        self.data = {}        # nested dict: {subj: {'video': ndarray}}
        self.samples = []     # list of (subj, start_idx)
        self.set_augmentations()
        self.load_dataset()
        

    def parse_subject(self, filename):
        filename = os.path.normpath(filename).split(os.sep)[-1]
        parts = os.path.normpath(filename).split("_")
        # print(parts)
        assert len(parts) >= 7, f"Path '{filename}' is too shallow. Expected format: data/OpenMV/001/850/L/Warm"
        cam = parts[-5]
        user = parts[-7]
        wavelength = parts[-3]
        hand = parts[-6]
        condition = parts[-2]
        clip_id = parts[-1].replace('.npy', '')
        return f"{user}_{hand}_{cam}_{wavelength}_{condition}_{clip_id}"

    def load_dataset(self):
        npy_files = sorted(glob.glob(os.path.join(self.data_path, "*.npy")))
        for path in npy_files:
            subj = self.parse_subject(path)
            clip = np.load(path)  # shape: [T, H, W, C]

            assert clip.shape[0] == self.frames_per_clip, \
                f"Clip {path} has {clip.shape[0]} frames, expected {self.frames_per_clip}"
            
            if any(str(w) in path for w in self.wavelength) and any(str(c) in path for c in self.cam) and any(str(c) in path for c in self.condition):
                self.samples.append((subj, clip))  # store each clip as one sample

        print(f"[DEBUG] Dataset size: {len(self.samples)}")

    def set_augmentations(self):
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        self.aug_reverse = False
        if self.split == 'train':
            self.aug_flip = 'f' in self.aug
            self.aug_illum = 'i' in self.aug
            self.aug_gauss = 'g' in self.aug
            self.aug_speed = 's' in self.aug
            self.aug_resizedcrop = 'c' in self.aug
            self.aug_reverse = 'r' in self.aug

    def apply_transformations(self, clip, subj, idcs, augment=True):
        # mean = clip.mean()
        # std = clip.std()
        # print(f"[DEBUG] Clip mean: {mean}, std: {std}")
        if augment:
            # print(f"[DEBUG] Applying augmentations to subject: {subj}, indices: {idcs}")
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)
            if self.aug_speed:
                # Add missing arguments here
                clip, idcs, _ = transforms.augment_speed(
                    clip,
                    idcs,
                    self.frames_per_clip,
                    self.channels,
                    self.speed_slow,
                    self.speed_fast
                )
            # print(f"[DEBUG] Clip shape after speed augmentation: {clip.shape}")
            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)
            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)
        # mean = clip.mean()
        # std = clip.std()
        # clip = (clip - mean) / (std + 1e-6)
        # print(f"[DEBUG] Clip mean after aug: {mean}, std: {std}")
        return clip, idcs, 1


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, clip = self.samples[idx]
        idcs = np.arange(0, self.frames_per_clip, dtype=int)
        clip = transforms.prepare_clip(clip, self.channels)  # [C, T, H, W]
        if self.type == 'static':
            # print("[DEBUG] Mimicing fake data...")
            clip = clip[:, :1, :, :]
            clip = np.repeat(clip, self.frames_per_clip, axis=1)
        elif self.type == 'shuffle':
            # Shuffle frame order
            indices = np.arange(self.frames_per_clip)
            np.random.shuffle(indices)
            clip = clip[:, indices, :, :]
        elif self.type == 'static_periodic':
            # Freeze spatial content
            clip = clip[:, :1, :, :]  # shape [C, 1, H, W]
            clip = np.repeat(clip, self.frames_per_clip, axis=1)  # [C, T, H, W]

            # Parameters
            fps = 30  # frames per second; update to match your data
            T = self.frames_per_clip
            C, _, H, W = clip.shape

            # Random pulse frequency in Hz (e.g., 0.8–2.5 Hz = 48–150 bpm)
            pulse_freq = np.random.uniform(0.8, 2.5)  # Hz
            t = np.linspace(0, T / fps, T)  # time in seconds

            # Create modulation: sinusoidal variation centered at 1
            delta = 0.05  # 5% modulation strength; adjust as needed
            modulation = 1.0 + delta * np.sin(2 * np.pi * pulse_freq * t)  # shape [T]

            # Apply to clip (broadcast to [C, T, H, W])
            modulation = modulation[None, :, None, None]  # reshape to [1, T, 1, 1]
            clip = clip * modulation.astype(np.float32)

        
        if self.split == 'train':
            clip, idcs, speed = self.apply_transformations(clip, subj, idcs)
        else:
            clip, idcs, speed = self.apply_transformations(clip, subj, idcs, augment=False)

        return torch.from_numpy(clip.astype(np.float32)), subj, idcs, speed
        
