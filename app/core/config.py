from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = BASE_DIR / "models"

VOCAB_DICT_DIR = MODELS_DIR / "vocab_dict.json"

EMOTION_MODEL_DIR = MODELS_DIR / "emotion_classifier.pth"

LANGUAGE_MODEL_PATH = MODELS_DIR / "language_detector.pkl"
