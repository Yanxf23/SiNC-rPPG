import torch
import torch.nn as nn
import imageio
import numpy as np
import os
import cv2
import sys

def save_clip_as_mp4(clip_tensor, save_path="clip.mp4", fps=10):
    """
    Save a [C, T, H, W] tensor as an MP4 video using OpenCV.
    """
    C, T, H, W = clip_tensor.shape
    assert C in [1, 3], f"Unexpected number of channels: {C}"

    out_h, out_w = int(H), int(W)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (out_w, out_h), isColor=True)

    if not writer.isOpened():
        raise RuntimeError(f"❌ Failed to open video writer for {save_path}")

    # Convert tensor to numpy array for processing
    clip_tensor = clip_tensor.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

    for t in range(T):
        frame = clip_tensor[t]  # [H, W, C] or [H, W, 1]

        # Normalize if needed and ensure uint8
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

        if C == 1:
            frame = np.squeeze(frame, axis=-1)     # [H, W]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # [H, W, 3]

        writer.write(frame)

    writer.release()
    print(f"✅ Saved MP4 video to {save_path}")
    sys.exit(0)

def debug_hook(module, input, output):
    save_clip_as_mp4(input[0][0].cpu())  # Show first clip in batch

class PhysNet(nn.Module):
    def __init__(self, input_channels=1, drop_p=0.5, t_kern=5, padding_mode='replicate'):
        '''
        input_channels: the number of channels of input video (RGB=3)
        drop_p: dropout probability during training
        t_kern: temporal kernel width
        padding_mode: pad for input and convolutions to avoid edge effects
        '''
        super(PhysNet, self).__init__()

        t_pad =  (t_kern//2, 1, 1)

        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(1,5,5), padding=(0,2,2), padding_mode=padding_mode)
        # self.conv1.register_forward_hook(debug_hook)
        self.bn1 = nn.BatchNorm3d(32)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(t_kern,3,3), padding=t_pad, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm3d(64)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn5 = nn.BatchNorm3d(64)
        self.max_pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn6 = nn.BatchNorm3d(64)

        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn7 = nn.BatchNorm3d(64)
        self.max_pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn8 = nn.BatchNorm3d(64)

        self.conv9 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), dilation=(1,1,1), padding=t_pad, padding_mode=padding_mode)
        self.bn9 = nn.BatchNorm3d(64)

        self.avg_pool1 = nn.AvgPool3d(kernel_size=(1,4,4), stride=(1,2,2))
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1)

        self.drop3d = nn.Dropout3d(drop_p)

        self.forward_stream = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.max_pool1,
            self.conv2, self.bn2, nn.ReLU(),
            self.conv3, self.bn3, nn.ReLU(), self.drop3d, self.max_pool2,
            self.conv4, self.bn4, nn.ReLU(),
            self.conv5, self.bn5, nn.ReLU(), self.drop3d, self.max_pool3,
            self.conv6, self.bn6, nn.ReLU(),
            self.conv7, self.bn7, nn.ReLU(), self.drop3d, self.max_pool4,
            self.conv8, self.bn8, nn.ReLU(),
            self.conv9, self.bn9, nn.ReLU(), self.drop3d, self.avg_pool1,
            self.conv10
        )

    def forward(self, x):
        # print("Input shape:", x.shape)         # Should be [B, C, T, H, W]
        x = self.forward_stream(x)
        x = torch.flatten(x, start_dim=1, end_dim=4)
        # print("Output shape:", x.shape)        # Should be [B, T]
        return x

