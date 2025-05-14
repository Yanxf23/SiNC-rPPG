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
        return f"{user}_{hand}_{cam}_{wavelength}_{condition}"

    def load_dataset(self):
        npy_files = sorted(glob.glob(os.path.join(self.data_path, "*.npy")))
        for path in npy_files:
            subj = self.parse_subject(path)
            clip = np.load(path)  # shape: [T, H, W, C]

            assert clip.shape[0] == self.frames_per_clip, \
                f"Clip {path} has {clip.shape[0]} frames, expected {self.frames_per_clip}"

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
        return clip, idcs, 1


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, clip = self.samples[idx]
        clip = transforms.prepare_clip(clip, self.channels)  # [C, T, H, W]

        if self.split == 'train':
            clip, idcs, speed = self.apply_transformations(clip, subj, 1)
        else:
            clip, idcs, speed = self.apply_transformations(clip, subj, 1, augment=False)

        return torch.from_numpy(clip.astype(np.float32)), subj, idcs, speed
        
