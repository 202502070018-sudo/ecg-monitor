"""Preprocessing shared with the Colab notebook (same filter, window and normalisation)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample

MODEL_FS = 360  # MIT-BIH sampling rate the model was trained at


def bandpass_filter(x, fs, low=0.5, high=40.0, order=3):
    nyq = fs / 2
    high = min(high, nyq * 0.95)
    low = min(low, high * 0.5)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def to_model_rate(x, fs):
    """Resample to 360 Hz so the window length matches training."""
    x = np.asarray(x, dtype=float)
    if int(fs) == MODEL_FS:
        return x
    return resample(x, int(round(len(x) * MODEL_FS / fs)))


def detect_r_peaks(filtered, fs=MODEL_FS):
    """Simple R-peak detector (the notebook used dataset annotations; uploads have none)."""
    sig = np.abs(filtered)
    thr = np.mean(sig) + 1.0 * np.std(sig)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.25 * fs))
    return peaks


def segment_beats(filtered, peaks, pre_s, post_s, fs=MODEL_FS):
    pre, post = int(pre_s * fs), int(post_s * fs)
    segs, kept = [], []
    for p in peaks:
        s, e = int(p) - pre, int(p) + post
        if s < 0 or e > len(filtered):
            continue
        beat = filtered[s:e]
        med = np.median(beat)
        mad = np.median(np.abs(beat - med)) + 1e-8
        segs.append(((beat - med) / (1.4826 * mad)).astype(np.float32))
        kept.append(p)
    return np.asarray(segs, dtype=np.float32), np.asarray(kept)
