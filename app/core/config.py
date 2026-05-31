import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
VOCAB_DICT_DIR = MODELS_DIR / "vocab_dict.json"
EMOTION_MODEL_DIR = MODELS_DIR / "emotion_classifier.pth"
LANGUAGE_MODEL_PATH = MODELS_DIR / "language_detector.pkl"

# LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Qdrant Database Configuration
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "mental_health_hybrid"