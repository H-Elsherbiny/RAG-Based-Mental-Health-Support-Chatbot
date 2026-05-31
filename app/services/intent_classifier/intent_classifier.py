import os
from groq import Groq
from dotenv import load_dotenv
from .prompts import FEW_SHOT_SYSTEM_PROMPT

load_dotenv(override=True)

class IntentClassifier:
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.0):
        """Initializes the Intent Classifier with the Groq client."""
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = FEW_SHOT_SYSTEM_PROMPT
        
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def classify_intent_few_shot(self, question: str) -> str:
        """Calls the LLM to classify the intent based on few-shot examples."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return response.choices[0].message.content.strip()

    def predict_intent(self, question: str) -> dict:
        """Main method to predict the intent and return a structured response."""
        intent = self.classify_intent_few_shot(question)
        return {
            "text": question,
            "intent": intent
        }

