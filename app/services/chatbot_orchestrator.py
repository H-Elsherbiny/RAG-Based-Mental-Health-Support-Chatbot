from groq import Groq
from app.core.config import GROQ_API_KEY
from app.services.intent_classifier import IntentClassifier
from app.services.emotion_classifier import EmotionClassifier
from app.services.language_detector import LanguageDetector
from app.services.rag import RAGPipeline

class ChatbotOrchestrator:
    def __init__(
        self, 
        intent_classifier: IntentClassifier,
        emotion_classifier: EmotionClassifier,
        language_detector: LanguageDetector,
        rag_pipeline: RAGPipeline
    ):
        """
        Initializes the orchestrator with all necessary NLP services via dependency injection.
        """
        self.intent_classifier = intent_classifier
        self.emotion_classifier = emotion_classifier
        self.language_detector = language_detector
        self.rag_pipeline = rag_pipeline
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    def _get_static_response(self, intent: str) -> str:
        """
        Helper method to return English static responses based on intent.
        """
        responses = {
            "greeting": "Hello! How can I support you today?",
            "goodbye": "Take care! Remember I'm always here if you need to talk.",
            "gratitude": "You're very welcome. I'm glad I could help.",
            "out_of_scope": "I am an AI specialized in mental health support. I cannot answer queries outside of this scope."
        }
        
        return responses.get(intent, "I don't understand.")

    def _translate_to_english(self, text: str, model_name: str = "openai/gpt-oss-20b") -> str:
        prompt = f"Translate the following text to English. Respond ONLY with the translation, nothing else.\n\nText: {text}"
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _translate_from_english(self, text: str, target_language: str, model_name: str = "openai/gpt-oss-20b") -> str:
        prompt = f"Translate the following text to {target_language}. Respond ONLY with the translation, nothing else. Do not add any conversational filler.\n\nText: {text}"
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _generate_final_response(self, user_query: str, emotion: str, context: str, model_name: str = "openai/gpt-oss-20b") -> str:
        system_prompt = (
            "You are an empathetic, professional mental health support chatbot. "
            "Use the provided historical counselor advice (Context) to synthesize a helpful, supportive response. "
            "Do NOT mention that you are referencing documents or 'context' to the user. "
            "Acknowledge the user's emotion subtly if appropriate. "
            "Keep the response concise and actionable."
        )
        user_prompt = f"User's Emotion: {emotion}\n\nContext:\n{context}\n\nUser Query: {user_query}"
        
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def process_message(self, user_message: str) -> str:
        """
        Main pipeline that processes the user query and routes it to the corresponding logic.
        """
        # 1. Detect Language
        lang_result = self.language_detector.predict_language(user_message)
        language_code = lang_result['language_code']
        language_name = lang_result.get('language_name', 'Unknown')

        # 2. Translate to English if needed
        if language_code not in ["en", "unknown"]:
            english_message = self._translate_to_english(user_message)
        else:
            english_message = user_message

        # 3. Classify Intent (before emotion extraction to avoid unnecessary steps)
        intent_result = self.intent_classifier.predict_intent(english_message)
        intent = intent_result['intent']

        # 4. Route based on Intent
        if intent in ["greeting", "goodbye", "gratitude", "out_of_scope"]:
            english_response = self._get_static_response(intent)

        elif intent == "asking_mental_health_question":
            # 5. Extract Emotion (only for actual questions)
            emotion_result = self.emotion_classifier.predict_emotion(english_message)
            emotion = emotion_result['emotion']

            # 6. RAG Pipeline
            search_output = self.rag_pipeline.two_stage_hybrid_search(
                raw_user_query=english_message,
                top_k_scenarios=3,
                top_k_advice=5,
                top_n_final=2
            )
            context = self.rag_pipeline.format_retrieved_context(search_output)

            # 7. Generate Response using LLM
            english_response = self._generate_final_response(english_message, emotion, context)
            
        else:
            english_response = self._get_static_response("out_of_scope")

        # 8. Translate back if necessary
        if language_code not in ["en", "unknown"]:
            final_response = self._translate_from_english(english_response, language_name)
        else:
            final_response = english_response

        return final_response
