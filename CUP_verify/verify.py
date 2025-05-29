import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
from scipy.signal import periodogram, find_peaks, windows
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy.fft import fft
from scipy.signal import welch

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_root", type=str, required=True)
parser.add_argument("--condition", type=str, required=True)
parser.add_argument("--cam", type=str, default="See3", required=False) #See3, OpenMV
args = parser.parse_args()

experiment_root = args.experiment_root
condition = args.condition
cam = args.cam
directory = os.path.join(experiment_root, "predictions")
results_dir = os.path.join(experiment_root, "results", condition)
os.makedirs(results_dir, exist_ok=True)

# === Ground truth reader ===
def read_csv(user_id):
    csv_path = r"C:\Users\mobil\Desktop\25summer\PPGPalm\SiNC-rPPG\CUP_verify\user_info.csv"
    df = pd.read_csv(csv_path)
    user_row = df[df['User ID'].astype(str).str.zfill(3) == user_id]
    if not user_row.empty:
        ground_truth = {
            'pulse_rate_bpm': user_row['Pulse Rate'].values[0],
            'breaths_per_min': user_row['Breaths (min)'].values[0],
            'PVI': user_row['Pleth Variability'].values[0],
            'PI': user_row['Perfusion Index'].values[0]
        }
    else:
        ground_truth = {}
    return ground_truth

def load_npy(file_path):
    if os.path.exists(file_path):
        predicted_rppg = np.load(file_path)
        signal_length = len(predicted_rppg)
        duration = signal_length / 30
        t = np.linspace(0, duration, signal_length)
    else:
        predicted_rppg = None
        t = None
    return predicted_rppg, t

def estimate_pulse_rate(tmp_gt, file_path, sig, fs, plot=True, harmonics_removal=True, title=None):
    # Windowed and unwindowed versions
    sig_nowin = sig.copy()
    hann_win = signal.windows.hann(len(sig))
    sig_win = sig * hann_win

    def process_welch_and_peak(signal_input, fs, harmonics_removal):
        freq_axis, power = welch(signal_input, fs=fs, nperseg=min(64, len(signal_input)))

        # Bandpass range
        mask = (freq_axis >= 0.66) & (freq_axis <= 3.0)
        power_masked = np.zeros_like(power)
        power_masked[mask] = power[mask]

        # Find peaks
        peak_idx, _ = signal.find_peaks(power_masked)
        if len(peak_idx) < 1:
            return None, None, power, power_masked, freq_axis

        sorted_idx = np.argsort(power_masked[peak_idx])[::-1]
        peak_idx1 = peak_idx[sorted_idx[0]]
        peak_idx2 = peak_idx[sorted_idx[1]] if len(sorted_idx) > 1 else peak_idx1

        def parabolic_interpolation(y, k):
            if 1 <= k < len(y) - 1:
                left, center, right = y[k - 1], y[k], y[k + 1]
                delta = 0.5 * (left - right) / (left - 2 * center + right)
                interpolated_idx = k + delta
                return interpolated_idx
            return k

        # Apply interpolation
        interp_idx1 = parabolic_interpolation(power_masked, peak_idx1)
        interp_idx2 = parabolic_interpolation(power_masked, peak_idx2)

        freq_interp1 = freq_axis[0] + (freq_axis[1] - freq_axis[0]) * interp_idx1
        freq_interp2 = freq_axis[0] + (freq_axis[1] - freq_axis[0]) * interp_idx2
        hr1 = freq_interp1 * 60
        hr2 = freq_interp2 * 60

        if harmonics_removal and abs(hr1 - 2 * hr2) < 10:
            hr = hr2
            selected_peak = interp_idx2
        else:
            hr = hr1
            selected_peak = interp_idx1

        return hr, selected_peak, power, power_masked, freq_axis

    # Process both signals
    hr_nowin, peak_nowin, Pxx_nowin, Pxx_nowin_masked, freq_axis = process_welch_and_peak(sig_nowin, fs, harmonics_removal)
    hr_win, peak_win, Pxx_win, Pxx_win_masked, _ = process_welch_and_peak(sig_win, fs, harmonics_removal)

    hr = hr_win if hr_win is not None else hr_nowin  # Prefer windowed result

    def plot_welch_comparison_interpolated(
        sig_nowin, sig_win,
        freq_axis, Pxx_nowin, Pxx_nowin_masked, peak_nowin, hr_nowin,
        Pxx_win, Pxx_win_masked, peak_win, hr_win,
        tmp_gt, file_path, results_dir, title=None
    ):

        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        axs[0, 0].plot(sig_nowin)
        axs[0, 0].set_title("Raw Signal")
        axs[0, 0].set_xlabel("Time Index")
        axs[0, 0].set_ylabel("Amplitude")

        axs[0, 1].plot(sig_win)
        axs[0, 1].set_title("Signal with Hann Window")
        axs[0, 1].set_xlabel("Time Index")
        axs[0, 1].set_ylabel("Amplitude")

        # Interpolated freq for no window
        if peak_nowin is not None:
            freq_nowin_interp = freq_axis[0] + (freq_axis[1] - freq_axis[0]) * peak_nowin
        else:
            freq_nowin_interp = None

        axs[1, 0].plot(freq_axis, Pxx_nowin, alpha=0.4, label="Welch (No Window)")
        axs[1, 0].plot(freq_axis, Pxx_nowin_masked, linewidth=2, label="Masked")
        if freq_nowin_interp is not None:
            axs[1, 0].axvline(freq_nowin_interp, color='black', linestyle='--', label=f"Est: {hr_nowin:.2f} bpm")
        axs[1, 0].axvline(tmp_gt / 60, color='red', linestyle='--', label=f"GT: {tmp_gt:.2f} bpm")
        axs[1, 0].set_xlim(0, 6)
        axs[1, 0].set_xlabel("Frequency (Hz)")
        axs[1, 0].set_ylabel("Power")
        axs[1, 0].set_title("Welch Spectrum (No Window)")
        axs[1, 0].legend()

        # Interpolated freq for Hann window
        if peak_win is not None:
            freq_win_interp = freq_axis[0] + (freq_axis[1] - freq_axis[0]) * peak_win
        else:
            freq_win_interp = None

        axs[1, 1].plot(freq_axis, Pxx_win, alpha=0.4, label="Welch (Hann)")
        axs[1, 1].plot(freq_axis, Pxx_win_masked, linewidth=2, label="Masked")
        if freq_win_interp is not None:
            axs[1, 1].axvline(freq_win_interp, color='black', linestyle='--', label=f"Est: {hr_win:.2f} bpm")
        axs[1, 1].axvline(tmp_gt / 60, color='red', linestyle='--', label=f"GT: {tmp_gt:.2f} bpm")
        axs[1, 1].set_xlim(0, 6)
        axs[1, 1].set_xlabel("Frequency (Hz)")
        axs[1, 1].set_ylabel("Power")
        axs[1, 1].set_title("Welch Spectrum (Hann)")
        axs[1, 1].legend()

        plt.tight_layout()
        title = os.path.basename(file_path).split(".")[0] + "_welch" if title is None else title
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"{title}.png"))
        plt.close()
    
    if plot:
        plot_welch_comparison_interpolated(
            sig_nowin, sig_win,
            freq_axis, Pxx_nowin, Pxx_nowin_masked, peak_nowin, hr_nowin,
            Pxx_win, Pxx_win_masked, peak_win, hr_win,
            tmp_gt, file_path, results_dir, title
        )

    return hr, freq_axis, Pxx_win_masked

def estimate_breaths(signal, fs):
    b, a = sps.butter(2, [0.1 / (fs/2), 0.4 / (fs/2)], btype='band')
    filtered = sps.filtfilt(b, a, signal)
    f, Pxx = sps.periodogram(filtered, fs)
    peak_freq = f[np.argmax(Pxx)]
    return peak_freq * 60, filtered

def estimate_PI(signal):
    ac = np.std(signal)
    dc = np.mean(np.abs(signal))
    return (ac / dc) * 100

def estimate_PVI(signal, fs):
    peaks, _ = sps.find_peaks(signal, distance=fs/2.5)
    amplitudes = signal[peaks]
    return (np.std(amplitudes) / np.mean(amplitudes)) * 100 if len(amplitudes) > 1 else 0

def get_results(file_path):
    user_id = file_path.split("\\")[-1].split("_")[0]
    rppg_signal, t = load_npy(file_path)
    fs = 30
    ground_truth = read_csv(user_id)
    tmp_gt = ground_truth.get("pulse_rate_bpm", np.nan)
    if np.isnan(tmp_gt):
        print(f"Warning: No ground truth data for {user_id}. Skipping pulse rate estimation.")
        pulse_est, f_hr, Pxx_hr = -1, -1, -1
    else:
        pulse_est, f_hr, Pxx_hr = estimate_pulse_rate(tmp_gt, file_path, rppg_signal, fs)
    breaths_est, filtered_breath = estimate_breaths(rppg_signal, fs)
    pi_est = estimate_PI(rppg_signal)
    pvi_est = estimate_PVI(rppg_signal, fs)
    results = {
        "User ID": user_id,
        "Estimated Pulse Rate (bpm)": pulse_est,
        "Ground Truth Pulse Rate (bpm)": ground_truth.get("pulse_rate_bpm", np.nan),
        "Estimated Breaths/min": breaths_est,
        "Ground Truth Breaths/min": ground_truth.get("breaths_per_min", np.nan),
        "Estimated PI": pi_est,
        "Ground Truth PI": ground_truth.get("PI", np.nan),
        "Estimated PVI": pvi_est,
        "Ground Truth PVI": ground_truth.get("PVI", np.nan)
    }
    # print("user number from csv: ", len(results))
    return results

def process_directory(directory, condition, cam):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                if condition not in file or cam not in file:
                    continue
                file_path = os.path.join(root, file)
                results.append(get_results(file_path))
    return results

def plot_comparison_grouped(df, signal_name, est_col, gt_col, save_path):
    print(df)
    df_grouped = df.groupby('User ID')[[gt_col, est_col]].agg(['mean', 'std'])
    users = df_grouped.index.tolist()
    gt_mean = df_grouped[(gt_col, 'mean')]
    gt_std = df_grouped[(gt_col, 'std')]
    est_mean = df_grouped[(est_col, 'mean')]
    est_std = df_grouped[(est_col, 'std')]

    x = range(len(users))
    plt.figure(figsize=(16, 4))
    plt.errorbar(x, gt_mean, yerr=gt_std, fmt='o-', label='Ground Truth')
    plt.errorbar(x, est_mean, yerr=est_std, fmt='s--', label='Estimated')

    plt.xticks(x, users, rotation=45)
    plt.ylabel(signal_name)
    plt.title(f"{signal_name}: Estimated vs Ground Truth (Grouped by User)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison(df, signal_name, est_col, gt_col, save_path):
    plt.figure(figsize=(8, 4))
    x = range(len(df))
    # print(df.columns)
    plt.plot(x, df[gt_col], 'o-', label='Ground Truth')
    plt.plot(x, df[est_col], 's--', label='Estimated')
    plt.xticks(x, df['User ID'], rotation=45)
    plt.ylabel(signal_name)
    plt.title(f"{signal_name}: Estimated vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_comparison_new(df, signal_name, est_col, gt_col, save_path): 
    plt.figure(figsize=(6, 6))
    
    x = df[gt_col]
    y = df[est_col]
    print(df, x, y)
    
    plt.scatter(x, y, c='blue', marker='o', label='Data Points')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='Ideal')  # Identity line
    
    plt.xlabel(f"Ground Truth {signal_name}")
    plt.ylabel(f"Estimated {signal_name}")
    plt.title(f"{signal_name}: Estimated vs Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === Run and evaluate ===
results = process_directory(directory, args.condition, args.cam)
df = pd.DataFrame(results)
df.columns = df.columns.str.strip()
df = df.dropna()

# --- Save plots ---
# print("Available columns in df:", df.columns.tolist())
plot_comparison_new(df, "Pulse Rate (bpm)", "Estimated Pulse Rate (bpm)", "Ground Truth Pulse Rate (bpm)", os.path.join(results_dir, "pulse_rate.png"))
# plot_comparison(df, "Breathing Rate (bpm)", "Estimated Breaths/min", "Ground Truth Breaths/min", os.path.join(results_dir, "breathing_rate.png"))
# plot_comparison(df, "Perfusion Index", "Estimated PI", "Ground Truth PI", os.path.join(results_dir, "pi.png"))
# plot_comparison(df, "Pleth Variability Index", "Estimated PVI", "Ground Truth PVI", os.path.join(results_dir, "pvi.png"))

# plot_comparison_grouped(df, "Pulse Rate (bpm)", "Estimated Pulse Rate (bpm)", "Ground Truth Pulse Rate (bpm)", os.path.join(results_dir, "pulse_rate_new.png"))

# --- Save metrics ---
def compute_metrics(gt, est):
    gt, est = np.array(gt).astype(float), np.array(est).astype(float)
    mask = np.isfinite(gt) & np.isfinite(est)
    if mask.sum() == 0:
        return None
    gt, est = gt[mask], est[mask]
    mae = mean_absolute_error(gt, est)
    rmse = root_mean_squared_error(gt, est)
    
    return mae, rmse

def compute_metrics_per_user(df, user_col, gt_col, est_col):
    user_errors = {}
    for user_id, group in df.groupby(user_col):
        gt = group[gt_col]
        est = group[est_col]
        result = compute_metrics(gt, est)
        if result is not None:
            mae, rmse = result
            # print(f"User {user_id}: MAE={mae:.2f}, RMSE={rmse:.2f}")
            user_errors[user_id] = (mae, rmse)
        else:
            # print(f"User {user_id}: invalid or missing data")
            user_errors[user_id] = None
    return user_errors

metrics = {
    "Pulse Rate (bpm)": compute_metrics(df["Ground Truth Pulse Rate (bpm)"], df["Estimated Pulse Rate (bpm)"]),
    "Breathing Rate (bpm)": compute_metrics(df["Ground Truth Breaths/min"], df["Estimated Breaths/min"]),
    "Perfusion Index": compute_metrics(df["Ground Truth PI"], df["Estimated PI"]),
    "Pleth Variability Index": compute_metrics(df["Ground Truth PVI"], df["Estimated PVI"])
}

user_errors = compute_metrics_per_user(df, user_col='User ID', gt_col='Ground Truth Pulse Rate (bpm)', est_col='Estimated Pulse Rate (bpm)')
with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
    for signal, m in metrics.items():
        f.write(f"{signal}:\n")
        if m is None:
            f.write("  No valid data.\n\n")
        else:
            mae, rmse = m
            f.write(f"  MAE  = {mae:.2f}\n")
            f.write(f"  RMSE = {rmse:.2f}\n")
    f.write("\n\n")
    for user_id, m in user_errors.items():
        mae, rmse = m
        f.write(f"User {user_id} MAE: {mae:.2f}\n")

