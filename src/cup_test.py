import os
import torch
import numpy as np
import argparse
from utils.model_selector import select_model
from utils import validate as validate_utils
import sys
sys.path.append(r"./datasets")
from db_utils import get_dataset
import args
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # --- Inference configuration ---
    experiment_dir = arg_obj.experiment_root  # e.g., experiments/cup_model/
    model_dir = os.path.join(experiment_dir, 'best_saved_models')
    model_tag = os.listdir(model_dir)[0]
    model_path = os.path.join(model_dir, model_tag)
    print(f"Using model: {model_path}")

    # --- Output folder setup ---
    output_dir = os.path.join('../predictions', experiment_dir.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)

    # --- Load test dataset ---
    arg_obj.dataset = 'cup_unsupervised'
    arg_obj.split = 'val'
    test_set = get_dataset(arg_obj.split, arg_obj)

    # --- Load model ---
    arg_obj.model_type = model_tag.split('_')[0]  # assumes model type is first token
    model = select_model(arg_obj)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float().to(device)
    model.eval()

    # --- Inference (CUP: no ground truth, just predictions) ---
    print("Running inference on CUP test set...")
    pred_waves, identifiers = validate_utils.infer_over_dataset_testing_cup(
        model, test_set, criterion=None, device=device, args=arg_obj)

    for wave, name in zip(pred_waves, identifiers):
        save_path = os.path.join(output_dir, "predictions", f"{name}_wave.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, wave)
        plot_waveform(save_path, save=True)

    print(f"✅ Saved {len(pred_waves)} predicted waveforms to {output_dir}")

if __name__ == "__main__":
    main()
