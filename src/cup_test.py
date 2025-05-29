import os
import torch
import numpy as np
import argparse
from utils.model_selector import select_model
from utils import validate as validate_utils
from scipy.fft import rfft, rfftfreq
import sys
sys.path.append(r"./datasets")
from db_utils import get_dataset
import args
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_spectrum(signal, fs=30, label='', color='blue'):
    f = rfftfreq(len(signal), 1/fs)
    spectrum = np.abs(rfft(signal))
    plt.plot(f, spectrum, label=label, color=color)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

def plot_mask_gif(raw_video, early_mask=None, mid_mask=None, save_path="output.gif", fps=5):
    """
    Creates a grayscale side-by-side GIF: [Raw | Early Mask | Mid Mask].

    Args:
        raw_video: np.ndarray (C, T, H, W) or (T, H, W)
        early_mask: np.ndarray (T, H, W) or None
        mid_mask: np.ndarray (T, H, W) or None
        save_path: path to output .gif file
        fps: frames per second
    """
    if early_mask is not None and early_mask.ndim == 4 and early_mask.shape[0] == 1:
        early_mask = early_mask.squeeze(0)
    if mid_mask is not None and mid_mask.ndim == 4 and mid_mask.shape[0] == 1:
        mid_mask = mid_mask.squeeze(0)

    if raw_video.ndim == 4:  # (C, T, H, W)
        C, T, H, W = raw_video.shape
        frames = [raw_video[:, t].transpose(1, 2, 0).mean(axis=-1) for t in range(T)]
    elif raw_video.ndim == 3:  # (T, H, W)
        T, H, W = raw_video.shape
        frames = [raw_video[t] for t in range(T)]
    else:
        raise ValueError("Unsupported raw video shape")

    images = []
    for t in range(T):
        row = []

        # Raw frame
        raw_frame = frames[t]
        raw_norm = (raw_frame - raw_frame.min()) / (raw_frame.ptp() + 1e-6)
        raw_uint8 = (255 * raw_norm).astype(np.uint8)
        row.append(raw_uint8)

        # Early mask
        if early_mask is not None:
            m = early_mask[t]
            m_norm = (m - m.min()) / (m.ptp() + 1e-6)
            m_uint8 = (255 * m_norm).astype(np.uint8)
            # m_uint8 = (255 * m).astype(np.uint8)
            row.append(m_uint8)

        # Mid mask
        # if mid_mask is not None:
        #     m = mid_mask[t]
        #     m_norm = (m - m.min()) / (m.ptp() + 1e-6)
        #     m_uint8 = (255 * m_norm).astype(np.uint8)
        #     # m_uint8 = (255 * m).astype(np.uint8)
        #     row.append(m_uint8)

        if mid_mask is not None:
            m = mid_mask[0, t]
            # print(f"mid_mask shape: {m.shape}")

            # Upsample to match raw frame size
            m_resized = cv2.resize(m, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

            # Normalize and convert to uint8
            m_norm = (m_resized - m_resized.min()) / (m_resized.ptp() + 1e-6)
            m_uint8 = (255 * m_norm).astype(np.uint8)
            row.append(m_uint8)

        # print(early_mask.shape, raw_video.shape, t)
        # print(row[0].shape, row[1].shape, row[2].shape, len(row), t)
        combined = np.concatenate(row, axis=1)
        images.append(combined)

    imageio.mimsave(save_path, images, fps=fps)
    
def plot_waveform(npy_path, save=True):
    wave = np.load(npy_path)
    name = os.path.basename(npy_path).replace('.npy', '')

    plt.figure(figsize=(10, 4))
    plt.plot(wave, color='darkred', linewidth=1)
    plt.title(f"Predicted rPPG Waveform: {name}")
    plt.xlabel("Frame")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)

    if save:
        out_path = npy_path.replace('.npy', '.png')
        plt.savefig(out_path)
        # print(f"✅ Saved waveform plot to {out_path}")
    plt.close()

def main():
    arg_obj = args.get_input()
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument("--gif", type=str, default="False")
    local_args, _ = local_parser.parse_known_args()

    # --- Inference configuration ---
    experiment_dir = arg_obj.experiment_root  # e.g., experiments/cup_model/
    model_dir = os.path.join(experiment_dir, 'best_saved_models')
    model_tag = os.listdir(model_dir)[0]
    model_path = os.path.join(model_dir, model_tag)
    print(f"Using model: {model_path}")

    # --- Output folder setup ---
    output_dir = experiment_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load test dataset ---
    arg_obj.dataset = 'cup_unsupervised'
    arg_obj.split = 'val'
    test_set = get_dataset(arg_obj.split, arg_obj)

    # --- Load model ---
    arg_obj.model_type = model_tag.split('_')[0]  # assumes model type is first token
    model = select_model(arg_obj)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float().to(device)
    model.eval()

    # --- Inference (CUP: no ground truth, just predictions) ---
    print("Running inference on CUP test set...")
    pred_waves, identifiers, raw_inputs, attention_masks, unmasked_signals = validate_utils.infer_over_dataset_testing_cup(
        model, test_set, criterion=None, device=device, args=arg_obj)

    # for wave, name in zip(pred_waves, identifiers):
    #     save_path = os.path.join(output_dir, "predictions", f"{name}_wave.npy")
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     np.save(save_path, wave)
    #     # plot_waveform(save_path, save=True)

    for wave, name, raw, mask, unmasked in zip(pred_waves, identifiers, raw_inputs, attention_masks, unmasked_signals):
        save_path = os.path.join(output_dir, "predictions", f"{name}_wave.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, wave)

        # --- Plot FFT spectra comparison ---
        plt.figure(figsize=(8, 4))
        plot_spectrum(unmasked, label="Unmasked", color='gray')
        plot_spectrum(wave, label="Masked", color='darkred')
        plt.title(f"FFT: {name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path.replace('_wave.npy', '_fft_comparison.png'))
        plt.close()

        gif_path = save_path.replace('_wave.npy', '_mask_overlay.gif')

        # Unpack early/mid masks from dict
        early = mask.get('early') if isinstance(mask, dict) else None
        mid = mask.get('mid') if isinstance(mask, dict) else None
        # print(early.min(), early.max(), early.mean())
        # print(mid.min(), mid.max(), mid.mean())
        # if local_args.gif.lower() == 'true':
        # plot_mask_gif(raw, early_mask=early, mid_mask=mid, save_path=gif_path, fps=15)

    print(f"✅ Saved {len(pred_waves)} predicted waveforms to {output_dir}")

if __name__ == "__main__":
    main()
