"""
Mapping between model output indices and reciter names.
The order must match the LabelEncoder used during training.
"""

CLASS_NAMES = {
    0: "AbdulBari_Althubaity",
    1: "AbdulRahman_Alsudais",
    2: "Abdullah_Albuaijan",
    3: "Ali_Alhothaify",
    4: "Bander_Balilah",
    5: "Maher_Almuaiqly",
    6: "Mohammed_Aluhaidan",
    7: "Mohammed_Ayoub",
    8: "Nasser_Alqutami",
    9: "Saad_Alghamdi",
    10: "Saud_Alshuraim",
    11: "Yasser_Aldossary",
}

NUM_CLASSES = len(CLASS_NAMES)


def get_class_name(index: int):
    """
    Returns the reciter name corresponding
    to the predicted class index.
    """
    return CLASS_NAMES.get(index, "Unknown")