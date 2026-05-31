import os
import joblib
import warnings
from .config import LANG_MAP

class LanguageDetector:
    """
    Inference class to classify the language of a text using Traditional ML.
    Based on the 'Traditional_ML (TF-IDF + SAGA LogReg)' pipeline from the notebook.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
        
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}. Please train and save the pipeline to a .pkl/.joblib file.")

    def predict_language(self, text: str) -> dict:
        """Main method to predict the language and return a structured response."""
        clean_text = str(text).strip()
        
        if not clean_text or self.pipeline is None:
            return {
                "text": text,
                "language_code": "unknown",
                "language_name": "Unknown"
            }

        # Predict using the sklearn pipeline
        prediction = self.pipeline.predict([clean_text])
        
        
        lang_code = str(prediction[0])
        lang_name = LANG_MAP.get(lang_code, "Unknown")
        
        return {
            "text": text,
            "language_code": lang_code,
            "language_name": lang_name
        }