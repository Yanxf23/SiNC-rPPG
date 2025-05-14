import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps  # safe alias

def read_csv(user_id):
    csv_path = r"C:\Users\mobil\Desktop\25summer\PPGPalm\SiNC-rPPG\experiments\user_info.csv"
    df = pd.read_csv(csv_path)

    user_row = df[df['User ID'].astype(str).str.zfill(3) == user_id]

    # Extract relevant ground truth values
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
    # Load the signal
    if os.path.exists(file_path):
        predicted_rppg = np.load(file_path)
        signal_length = len(predicted_rppg)
        duration = signal_length / 30  # Assuming 30 fps sampling
        t = np.linspace(0, duration, signal_length)
    else:
        predicted_rppg = None
        t = None
    
    return predicted_rppg, t

# 1. Estimate Pulse Rate (via FFT)
def estimate_pulse_rate(sig, fs):
    f, Pxx = sps.periodogram(sig, fs)
    mask = (f >= 0.66) & (f <= 3.0)
    f_valid, Pxx_valid = f[mask], Pxx[mask]
    peak_idx = np.argmax(Pxx_valid)
    peak_freq = f_valid[peak_idx]
    return peak_freq * 60, f_valid, Pxx_valid

# 2. Estimate Respiration Rate (via low-pass or envelope)
def estimate_breaths(signal, fs):
    b, a = sps.butter(2, [0.1 / (fs/2), 0.4 / (fs/2)], btype='band')
    filtered = sps.filtfilt(b, a, signal)
    f, Pxx = sps.periodogram(filtered, fs)
    peak_freq = f[np.argmax(Pxx)]
    return peak_freq * 60, filtered

# 3. Estimate PI (AC/DC ratio)
def estimate_PI(signal):
    ac = np.std(signal)
    dc = np.mean(np.abs(signal))
    return (ac / dc) * 100

# 4. Estimate PVI (amplitude variation over time)
def estimate_PVI(signal, fs):
    peaks, _ = sps.find_peaks(signal, distance=fs/2.5)
    amplitudes = signal[peaks]
    return (np.std(amplitudes) / np.mean(amplitudes)) * 100 if len(amplitudes) > 1 else 0

def get_results(file_path):
    user_id = file_path.split("\\")[-1].split("_")[0]

    rppg_signal, t = load_npy(file_path)
    fs = 30
    # Run estimations
    pulse_est, f_hr, Pxx_hr = estimate_pulse_rate(rppg_signal, fs)
    breaths_est, filtered_breath = estimate_breaths(rppg_signal, fs)
    pi_est = estimate_PI(rppg_signal)
    pvi_est = estimate_PVI(rppg_signal, fs)
    # Compile results

    ground_truth = read_csv(user_id)

    results = {
        "User ID": user_id,
        "Estimated Pulse Rate (bpm)": pulse_est,
        "Ground Truth Pulse Rate (bpm)": ground_truth["pulse_rate_bpm"],
        "Estimated Breaths/min": breaths_est,
        "Ground Truth Breaths/min": ground_truth["breaths_per_min"],
        "Estimated PI": pi_est,
        "Ground Truth PI": ground_truth["PI"],
        "Estimated PVI": pvi_est,
        "Ground Truth PVI": ground_truth["PVI"]
    }
    return results

# Function to plot one figure per physiological signal
def plot_comparison(df, signal_name, est_col, gt_col):
    plt.figure(figsize=(8, 4))
    x = range(len(df))
    plt.plot(x, df[gt_col], 'o-', label='Ground Truth')
    plt.plot(x, df[est_col], 's--', label='Estimated')
    plt.xticks(x, df['User ID'], rotation=45)
    plt.ylabel(signal_name)
    plt.title(f"{signal_name}: Estimated vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def process_directory(directory, condition, cam):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                if condition not in file:
                    continue
                if cam not in file:
                    continue
                file_path = os.path.join(root, file)
                result = get_results(file_path)
                results.append(result)
    return results

condition = "Clean"
cam = "See3"
directory = r"C:\Users\mobil\Desktop\25summer\PPGPalm\SiNC-rPPG\experiments\exper_0005\predictions"
results = process_directory(directory, condition, cam)
df = pd.DataFrame(results)

# Plot all four physiological signals
# plot_comparison(df, "Pulse Rate (bpm)", "Estimated Pulse Rate (bpm)", "Ground Truth Pulse Rate (bpm)")
# plot_comparison(df, "Breathing Rate (bpm)", "Estimated Breaths/min", "Ground Truth Breaths/min")
plot_comparison(df, "Perfusion Index", "Estimated PI", "Ground Truth PI")
# plot_comparison(df, "Pleth Variability Index", "Estimated PVI", "Ground Truth PVI")