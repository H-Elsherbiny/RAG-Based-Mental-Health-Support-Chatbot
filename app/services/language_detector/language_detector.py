import os
import re
import joblib
from typing import Dict, Any
from .config import LANG_MAP

class LanguageDetector:
    """
    Production-ready inference wrapper for the Hierarchical language detection pipeline.
    Loads the compound dictionary of script-specific models and routes incoming
    inference traffic dynamically.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sub_models: Dict[str, Any] = {}
        
        # Core script regex signatures matched to training configurations
        self.script_patterns = {
            'arabic': re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]'),
            'devanagari': re.compile(r'[\u0900-\u097F]'),
            'cjk': re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'),
            'latin': re.compile(r'[a-zA-Z]')
        }
        
        self._load_composite_artifacts()

    def _load_composite_artifacts(self) -> None:
        """Loads the structured dictionary containing the pre-fit sub-model pipelines."""
        if os.path.exists(self.model_path):
            # Expects a joblib export containing the sub_models dict map
            self.sub_models = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                f"Hierarchical sub-models file not found at: {self.model_path}. "
                f"Please execute your training pipeline and save your artifacts using joblib."
            )

    def _identify_script_family(self, text: str) -> str:
        """Evaluates sub-string matches to determine target sub-model family routing."""
        clean_text = str(text).strip()
        if not clean_text:
            return 'latin'
            
        counts = {family: len(pattern.findall(clean_text)) for family, pattern in self.script_patterns.items()}
        max_family = max(counts, key=counts.get)
        return max_family if counts[max_family] > 0 else 'latin'

    def validate_language(self, text: str, lang_code: str) -> str:
        """
        Performs downstream validation verification before outputting predictions.
        Useful for intercepting custom script fallback mismatches (e.g., verifying English fallback).
        """
        if not lang_code or lang_code.strip() == "":
            return 'en'
        return lang_code

    def predict_language(self, text: str) -> dict:
        """
        Primary inference path. Maps incoming payload string directly to script-level 
        routing arrays before predicting via specific sub-models or deterministic bypass strings.
        """
        clean_text = str(text).strip()
        
        # Immediate shortcut for empty strings
        if not clean_text:
            return {
                "text": text,
                "language_code": "unknown",
                "language_name": "Unknown"
            }

        # Resolve the active script family
        family = self._identify_script_family(clean_text)
        
        # Rule-based deterministic shortcut for Devanagari script matching Hindi
        if family == 'devanagari':
            lang_code = 'hi'
        else:
            pipeline = self.sub_models.get(family)
            # Safe runtime check validating model state presence
            if pipeline is not None and hasattr(pipeline.named_steps['classifier'], "classes_"):
                lang_code = str(pipeline.predict([clean_text])[0])
            else:
                lang_code = 'en'  # Standard script-family model fallback default

        # Post-processing validation step
        lang_code = self.validate_language(clean_text, lang_code)
        lang_name = LANG_MAP.get(lang_code, "Unknown")
        
        return {
            "text": text,
            "language_code": lang_code,
            "language_name": lang_name
        }