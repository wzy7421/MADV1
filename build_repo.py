import os
import zipfile

# ==========================================
# GitHub 项目文件内容定义
# ==========================================

readme_content = """# MAD Framework: Multi-Q Gabor Wavelet Perception

This repository contains the Python implementation of the Q-factor Wavelet Transform (QWT), Super-Resolution Averaged Projection (SRAP), and Minimum Intensity Projection (MinIP) used in the MAD autonomous driving perception framework.

## Project Structure
- `main.py`: The main execution script (equivalent to QCWTDemo2.m).
- `qwt_core.py`: Core algorithms for QWT, SRAP, and MinIP.
- `data_utils.py`: Signal preprocessing, FFT, and the Synthetic Signal Generator (from Appendix A).
- `requirements.txt`: Python dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo: `python main.py`
   *(If `data7.mat` is missing, the script will automatically generate the synthetic signal defined in the paper's Appendix A for validation).*
"""

requirements_content = """numpy
scipy
matplotlib
scipy.io
"""

qwt_core_content = """import numpy as np
from scipy.signal import morlet2, cwt

def compute_qwt(sig, fs, freqs, Q):
    \"\"\"
    Compute Q-factor Wavelet Transform using Morlet wavelet.
    \"\"\"
    w = 2.0 * Q
    # Convert frequencies to widths for scipy's cwt
    widths = w * fs / (2 * np.pi * freqs)
    cwt_matrix = cwt(sig, morlet2, widths, w=w)
    return np.abs(cwt_matrix)

def srap_fusion(qwt_list, weights=None):
    \"\"\"
    Super-Resolution Averaged Projection (SRAP).
    \"\"\"
    if weights is None:
        weights = np.ones(len(qwt_list)) / len(qwt_list)
    fused = np.zeros_like(qwt_list[0])
    for w, q_mat in zip(weights, qwt_list):
        fused += w * q_mat
    return fused

def minip_fusion(qwt_list):
    \"\"\"
    Super-resolution Minimum-Intensity Projection (MinIP).
    \"\"\"
    return np.min(np.array(qwt_list), axis=0)
"""

data_utils_content = """import numpy as np
from scipy.signal import butter, filtfilt, detrend

def preprocess_signal(sig, fs, lowcut=1.0, highcut=80.0):
    \"\"\"
    Zero-mean, Bandpass filter, Detrend, and Standardize.
    \"\"\"
    # Zero-mean
    sig = sig - np.mean(sig)

    # 4th-order Butterworth bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    sig = filtfilt(b, a, sig)

    # Detrend & Standardize
    sig = detrend(sig)
    sig = sig / np.std(sig)
    return sig

def compute_fft(sig, fs):
    \"\"\"
    Compute FFT for plotting.
    \"\"\"
    L = len(sig)
    NFFT = 2**int(np.ceil(np.log2(L)))
    Y = np.fft.rfft(sig, n=NFFT)
    f_fft = fs / 2 * np.linspace(0, 1, int(NFFT/2) + 1)
    mag_fft = np.abs(Y)
    return f_fft, mag_fft

def generate_synthetic_signal(fs=1024, duration=2.0):
    \"\"\"
    Generates the synthetic signal defined in Appendix A.
    \"\"\"
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 1. Chirp
    f0, f1 = 10, 150
    k = (f1 - f0) / duration
    s_chirp = 1.0 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # 2. Bursts (Transient events)
    s_burst = np.zeros_like(t)
    bursts = [(0.5, 0.05, 50), (1.2, 0.02, 120)]
    for ti, sigi, fi in bursts:
        s_burst += 1.5 * np.exp(-((t - ti)**2) / (2 * sigi**2)) * np.sin(2 * np.pi * fi * t)

    # 3. Stationary tones
    s_tone = 0.5 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 80 * t)

    # 4. Impulses
    s_imp = np.zeros_like(t)
    for tk in [0.3, 0.8, 1.5]:
        s_imp += 2.0 * np.exp(-((t - tk)**2) / (2 * 0.005**2))

    # 5. Noise (White + AR(1) Colored)
    nw = np.random.normal(0, 1, len(t))
    nc = np.zeros_like(t)
    rho = 0.8
    for i in range(1, len(t)):
        nc[i] = rho * nc[i-1] + np.random.normal(0, np.sqrt(1-rho**2))
    n = 0.2 * nw + 0.2 * nc

    x = s_chirp + s_burst + s_tone + s_imp + n
    return t, x
"""

main_content = """import os
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

    # Time Domain (Top spanning both columns)
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(xtime, sig, 'k', linewidth=1)
    ax_time.set_title(f"Time Domain Signal - {title_str}", fontsize=12, fontweight='bold')
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude (uV)")
    ax_time.set_xlim(xtime[0], xtime[-1])
    ax_time.grid(True, linestyle='--', alpha=0.6)

    # Time-Frequency QWT (Bottom Left)
    ax_qwt = fig.add_subplot(gs[1, 0])
    # shading='gouraud' matches MATLAB's shading interp
    pcm = ax_qwt.pcolormesh(xtime, freqs, cmap, shading='gouraud', cmap='jet', vmin=clim[0], vmax=clim[1])
    ax_qwt.set_title("QWT", fontsize=12)
    ax_qwt.set_xlabel("Time (s)")
    ax_qwt.set_ylabel("Freq (Hz)")
    ax_qwt.set_ylim(0, np.max(freqs))
    fig.colorbar(pcm, ax=ax_qwt, pad=0.02)

    # Frequency Domain (Bottom Right, vertical)
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
        # Assumes variable is named 'eeg'
        sig = np.squeeze(mat_data['eeg']).astype(np.float64)
        xtime = np.linspace(0, len(sig)/fs, len(sig), endpoint=False)
    else:
        print(f"Warning: {mat_path} not found. Generating paper's Synthetic Signal (Appendix A)...")
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
"""


# ==========================================
# 打包逻辑
# ==========================================

def create_project_zip():
    folder_name = "MAD_QWT_Framework"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    files = {
        "README.md": readme_content,
        "requirements.txt": requirements_content,
        "qwt_core.py": qwt_core_content,
        "data_utils.py": data_utils_content,
        "main.py": main_content
    }

    # 写入文件
    for filename, content in files.items():
        filepath = os.path.join(folder_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # 创建 ZIP 包
    zip_filename = f"{folder_name}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, project_files in os.walk(folder_name):
            for file in project_files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_name))
                zipf.write(file_path, arcname)

    print(f"✅ 成功生成项目压缩包: {zip_filename}")
    print(f"你可以直接将提取出的 {folder_name} 文件夹推送到 GitHub！")


if __name__ == "__main__":
    create_project_zip()