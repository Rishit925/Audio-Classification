import librosa
import numpy as np
import torch
from skimage.transform import resize


# ==========================
# Audio Preprocessing Config
# ==========================

SAMPLE_RATE = 22050
DURATION = 5

IMG_HEIGHT = 128
IMG_WIDTH = 256

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def get_spectrogram(file_path):
    """
    Convert an audio file into a normalized Mel Spectrogram.
    """

    # Load audio
    signal, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Generate Mel Spectrogram
    spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert Power Spectrogram to dB
    spec_db = librosa.power_to_db(
        spec,
        ref=np.max
    )

    # Z-score Normalization
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-8)

    # Fix Length
    spec_db = librosa.util.fix_length(
        spec_db,
        size=DURATION * SAMPLE_RATE // HOP_LENGTH + 1
    )

    # Resize
    spec_db = resize(
        spec_db,
        (IMG_HEIGHT, IMG_WIDTH),
        anti_aliasing=True
    )

    return spec_db.astype(np.float32)


def preprocess_audio(file_path):
    """
    Converts an audio file into a tensor
    ready for model inference.
    """

    spectrogram = get_spectrogram(file_path)

    spectrogram = torch.tensor(
        spectrogram,
        dtype=torch.float32
    )

    # Add Channel Dimension
    spectrogram = spectrogram.unsqueeze(0)

    # Add Batch Dimension
    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram