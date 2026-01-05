import numpy as np

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "The 'librosa' package is required for resample_audio. "
        "Install it with: pip install librosa"
    ) from e


def resample_audio(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio array x from orig_sr to target_sr using librosa.

    - x: 1D or 2D numpy array (we assume mono or last axis is time)
    - returns: 1D numpy array at target_sr
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        # fold to mono if needed
        x = np.mean(x, axis=-1)

    if orig_sr == target_sr:
        return x

    y = librosa.resample(y=x, orig_sr=orig_sr, target_sr=target_sr)
    return y.astype(np.float32)
