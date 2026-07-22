import torch
import torch.nn.functional as F

from app.model import load_model
from app.preprocess import preprocess_audio
from app.class_names import NUM_CLASSES, get_class_name
from app.config import MODEL_PATH


# Select Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Model Once
model = load_model(
    model_path=MODEL_PATH,
    num_classes=NUM_CLASSES,
    device=DEVICE
)


def predict(file_path):
    """
    Predict the Quran reciter from an audio file.

    Parameters
    ----------
    file_path : str
        Path to the uploaded audio file.

    Returns
    -------
    dict
        {
            "class_index": int,
            "reciter": str,
            "confidence": float
        }
    """

    # Preprocess Audio
    audio = preprocess_audio(file_path).to(DEVICE)

    # Disable Gradient Calculation
    with torch.no_grad():

        outputs = model(audio)

        probabilities = F.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

        class_index = predicted.item()
        confidence = confidence.item()

        return {
    "class_index": class_index,
    "reciter": get_class_name(class_index),
    "confidence": round(confidence * 100, 2)
}