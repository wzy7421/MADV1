import numpy as np
from scipy.signal import butter, filtfilt, detrend


def preprocess_signal(sig, fs, lowcut=1.0, highcut=80.0):
    """
    Zero-mean, Bandpass filter, Detrend, and Standardize.
    """
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
    """
    Compute FFT for plotting.
    """
    L = len(sig)
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    Y = np.fft.rfft(sig, n=NFFT)
    f_fft = fs / 2 * np.linspace(0, 1, int(NFFT / 2) + 1)
    mag_fft = np.abs(Y)
    return f_fft, mag_fft


def generate_synthetic_signal(fs=1024, duration=2.0):
    """
    Generates the synthetic signal defined in Appendix A of the MAD framework.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 1. Chirp
    f0, f1 = 10, 150
    k = (f1 - f0) / duration
    s_chirp = 1.0 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))

    # 2. Bursts (Transient events)
    s_burst = np.zeros_like(t)
    bursts = [(0.5, 0.05, 50), (1.2, 0.02, 120)]
    for ti, sigi, fi in bursts:
        s_burst += 1.5 * np.exp(-((t - ti) ** 2) / (2 * sigi ** 2)) * np.sin(2 * np.pi * fi * t)

    # 3. Stationary tones
    s_tone = 0.5 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 80 * t)

    # 4. Impulses
    s_imp = np.zeros_like(t)
    for tk in [0.3, 0.8, 1.5]:
        s_imp += 2.0 * np.exp(-((t - tk) ** 2) / (2 * 0.005 ** 2))

    # 5. Noise (White + AR(1) Colored)
    nw = np.random.normal(0, 1, len(t))
    nc = np.zeros_like(t)
    rho = 0.8
    for i in range(1, len(t)):
        nc[i] = rho * nc[i - 1] + np.random.normal(0, np.sqrt(1 - rho ** 2))
    n = 0.2 * nw + 0.2 * nc

    x = s_chirp + s_burst + s_tone + s_imp + n
    return t, x