import numpy as np
from scipy.signal import morlet2, cwt

def compute_qwt(sig, fs, freqs, Q):
    """
    Compute Q-factor Wavelet Transform using Morlet wavelet.
    """
    w = 2.0 * Q
    # Convert frequencies to widths for scipy's cwt
    widths = w * fs / (2 * np.pi * freqs)
    cwt_matrix = cwt(sig, morlet2, widths, w=w)
    return np.abs(cwt_matrix)

def srap_fusion(qwt_list, weights=None):
    """
    Super-Resolution Averaged Projection (SRAP).
    """
    if weights is None:
        weights = np.ones(len(qwt_list)) / len(qwt_list)
    fused = np.zeros_like(qwt_list[0])
    for w, q_mat in zip(weights, qwt_list):
        fused += w * q_mat
    return fused

def minip_fusion(qwt_list):
    """
    Super-resolution Minimum-Intensity Projection (MinIP).
    """
    return np.min(np.array(qwt_list), axis=0)