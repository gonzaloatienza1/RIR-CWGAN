import numpy as np
from scipy.signal import correlate, correlation_lags
from typing import Tuple

def align_rirs(rir1: np.ndarray, rir2: np.ndarray, target_samples: int = 6000, lag_threshold_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Align two Room Impulse Responses (RIRs) using cross-correlation, and match their
    length through truncation or zero-padding.

    Args:
        rir1 (np.ndarray): First RIR 1D array. Expected shape: (N,), where:
            
            - **N**: Number of samples in the RIR.

        rir2 (np.ndarray): Second RIR 1D array. Expected shape: (N,), where:

            - **N**: Number of samples in the RIR.

        target_samples (int, optional): Desired final length for both RIRs. Defaults to 6000.
        
        lag_threshold_ratio (float, optional): Max acceptable lag as a fraction of rir1's length. Defaults to 0.2.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
            - Aligned rir1 (np.ndarray) 1D array of shape (target_samples,)
            - Aligned rir2 (np.ndarray) 1D array of shape (target_samples,)
            - lag_max (int): The lag used to align the signals.
    """
    # Compute cross-correlation and lags
    corr = correlate(rir1, rir2, mode='full')
    lags = correlation_lags(len(rir1), len(rir2), mode='full')
    lag_max = lags[np.argmax(corr)]
    lag_threshold = int(lag_threshold_ratio * len(rir1))

    # If lag is too large, skip alignment
    if abs(lag_max) > lag_threshold:
        aligned_rir1 = rir1
        aligned_rir2 = rir2
    else:
        if lag_max > 0:
            aligned_rir1 = rir1[lag_max:]
            aligned_rir2 = rir2[:len(aligned_rir1)]
        elif lag_max < 0:
            aligned_rir2 = rir2[-lag_max:]
            aligned_rir1 = rir1[:len(aligned_rir2)]
        else:
            aligned_rir1 = rir1
            aligned_rir2 = rir2

        # Truncate to the same length
        min_len = min(len(aligned_rir1), len(aligned_rir2))
        aligned_rir1 = aligned_rir1[:min_len]
        aligned_rir2 = aligned_rir2[:min_len]

    # Pad or truncate to target length
    if len(aligned_rir1) < target_samples:
        pad_width = target_samples - len(aligned_rir1)
        aligned_rir1 = np.pad(aligned_rir1, (0, pad_width), mode='constant')
        aligned_rir2 = np.pad(aligned_rir2, (0, pad_width), mode='constant')
    else:
        aligned_rir1 = aligned_rir1[:target_samples]
        aligned_rir2 = aligned_rir2[:target_samples]

    return aligned_rir1, aligned_rir2, lag_max
