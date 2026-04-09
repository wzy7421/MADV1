import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from qwt_core import compute_qwt, srap_fusion, minip_fusion
from data_utils import preprocess_signal, compute_fft, generate_synthetic_signal


def plot_qcwt_figure(title_str, cmap, xtime, freqs, clim, fs, sig, f_fft, mag_fft):
    fig = plt.figure(figsize=(12, 8), num=title_str)
    fig.patch.set_facecolor('white')

    # Grid layout matching MATLAB's tiledlayout(2,2)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[4, 1], hspace=0.3, wspace=0.1)

    # Time Domain
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(xtime, sig, 'k', linewidth=1)
    ax_time.set_title(f"Time Domain Signal - {title_str}", fontsize=12, fontweight='bold')
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude (uV)")
    ax_time.set_xlim(xtime[0], xtime[-1])
    ax_time.grid(True, linestyle='--', alpha=0.6)

    # Time-Frequency QWT
    ax_qwt = fig.add_subplot(gs[1, 0])
    pcm = ax_qwt.pcolormesh(xtime, freqs, cmap, shading='gouraud', cmap='jet', vmin=clim[0], vmax=clim[1])
    ax_qwt.set_title("QWT", fontsize=12)
    ax_qwt.set_xlabel("Time (s)")
    ax_qwt.set_ylabel("Freq (Hz)")
    ax_qwt.set_ylim(0, np.max(freqs))
    fig.colorbar(pcm, ax=ax_qwt, pad=0.02)

    # Frequency Domain
    ax_fft = fig.add_subplot(gs[1, 1], sharey=ax_qwt)
    ax_fft.plot(mag_fft, f_fft, 'r', linewidth=1.2)
    ax_fft.set_title("Time-Frequency Signal", fontsize=12)
    ax_fft.set_xlabel("Amplitude")
    ax_fft.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    # 1. Parameter Configuration
    low_freq = 0
    high_freq = 250
    fs = 1024
    clim_zoom = 8
    Qmorlet = 5 / (2 * np.sqrt(2 * np.log(2)))
    freqs = np.arange(max(low_freq, 1), high_freq + 1)

    # 2. Data Loading (Fallback to Synthetic if MAT not found)
    mat_path = 'data7.mat'
    if os.path.exists(mat_path):
        print(f"Loading data from {mat_path}...")
        mat_data = sio.loadmat(mat_path)
        sig = np.squeeze(mat_data['eeg']).astype(np.float64)
        xtime = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
    else:
        print(f"Warning: {mat_path} not found. Generating Synthetic Signal (Appendix A)...")
        xtime, sig = generate_synthetic_signal(fs=fs, duration=2.0)

    # 3. Preprocessing
    sig_pre = preprocess_signal(sig, fs, lowcut=1.0, highcut=80.0)

    # 4. FFT
    f_fft, mag_fft = compute_fft(sig_pre, fs)

    # 5. Compute QCWT Mappings
    print("Computing Basic Morlet QCWT...")
    cmap_morlet = compute_qwt(sig_pre, fs, freqs, Qmorlet)

    print("Computing Multi-Q representations...")
    q_factors = Qmorlet * np.array([1, 20])
    qwt_list = [compute_qwt(sig_pre, fs, freqs, q) for q in q_factors]

    print("Applying SRAP and MinIP Fusions...")
    cmap_avg = srap_fusion(qwt_list)
    cmap_minip = minip_fusion(qwt_list)

    clim = [0, np.max(cmap_morlet) / clim_zoom]

    # 6. Plotting
    print("Rendering plots...")
    plot_qcwt_figure('Basic Morlet QCWT', cmap_morlet, xtime, freqs, clim, fs, sig_pre, f_fft, mag_fft)
    plot_qcwt_figure('Super-resolution Avg Projection', cmap_avg, xtime, freqs, clim, fs, sig_pre, f_fft, mag_fft)
    plot_qcwt_figure('Super-resolution MinIP Projection', cmap_minip, xtime, freqs, clim, fs, sig_pre, f_fft, mag_fft)


if __name__ == '__main__':
    main()