from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="Rishit925/Audio-Classification",
    filename="best_model.pth"
)

SAMPLE_RATE = 22050
DURATION = 5

IMG_HEIGHT = 128
IMG_WIDTH = 256

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128