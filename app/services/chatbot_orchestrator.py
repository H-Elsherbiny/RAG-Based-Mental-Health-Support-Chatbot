from groq import Groq
from langfuse import observe

from app.core.config import GROQ_API_KEY
from app.services.intent_classifier import IntentClassifier
from app.services.emotion_classifier import EmotionClassifier
from app.services.chat_history_store import ChatHistoryStore, InMemoryChatHistoryStore
from app.services.language_detector import LanguageDetector
from app.services.rag import RAGPipeline

class ChatbotOrchestrator:
    def __init__(
        self, 
        intent_classifier: IntentClassifier,
        emotion_classifier: EmotionClassifier,
        language_detector: LanguageDetector,
        rag_pipeline: RAGPipeline,
        history_store: ChatHistoryStore | None = None,
        max_history_turns: int = 5
    ):
        """
        Initializes the orchestrator with all necessary NLP services via dependency injection.
        """
        self.intent_classifier = intent_classifier
        self.emotion_classifier = emotion_classifier
        self.language_detector = language_detector
        self.rag_pipeline = rag_pipeline
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.history_store = history_store or InMemoryChatHistoryStore()
        self.max_history_messages = max_history_turns * 2

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

    @observe(name="translate_to_english", as_type="generation")
    def _translate_to_english(self, text: str, model_name: str = "openai/gpt-oss-20b") -> str:
        prompt = f"Translate the following text to English. Respond ONLY with the translation, nothing else.\n\nText: {text}"
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    @observe(name="translate_from_english", as_type="generation")
    def _translate_from_english(self, text: str, target_language: str, model_name: str = "openai/gpt-oss-20b") -> str:
        prompt = f"Translate the following text to {target_language}. Respond ONLY with the translation, nothing else. Do not add any conversational filler.\n\nText: {text}"
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    @observe(name="generate_final_response", as_type="generation")
    def _generate_final_response(self, user_query: str, emotion: str, context: str, chat_history: list = None, model_name: str = "openai/gpt-oss-20b") -> str:
        system_prompt = (
            "You are an empathetic, professional mental health support chatbot. "
            "Use the provided historical counselor advice (Context) to synthesize a helpful, supportive response. "
            "Do NOT mention that you are referencing documents or 'context' to the user. "
            "Acknowledge the user's emotion subtly if appropriate. "
            "Keep the response concise and actionable."
        )
        system_prompt = (
            "You are an empathetic, professional mental health support chatbot. "
            "Your goal is to provide supportive, actionable advice based primarily on the retrieved counselor knowledge. "
            "Strict Guidelines:\n"
            "1. Do NOT mention that you are referencing documents, 'context', or 'historical advice' to the user.\n"
            "2. Acknowledge the user's emotion gently and validate their feelings.\n"
            "3. Keep the response concise. Use bullet points for actionable steps if applicable.\n"
            "4. SAFETY CRITICAL: You are an AI, not a doctor. Do not diagnose conditions or recommend medications. "
            "If the user indicates severe crisis or self-harm, prioritize safety and advise seeking immediate professional emergency help.\n"
            "5. If the provided context is completely irrelevant to the user's query, rely on general empathetic support."
        )
        user_prompt = f"User's Emotion: {emotion}\n\nContext:\n{context}\n\nUser Query: {user_query}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            # Append the last N messages to maintain context
            for msg in chat_history:
                # Expecting msg to be a dict like {"role": "user"|"assistant", "content": "..."}
                messages.append(msg)
                
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    @observe(name="mental_health_chat")
    def process_message(self, user_message: str, session_id: str = "default") -> str:
        """
        Main pipeline that processes the user query and routes it to the corresponding logic.
        Maintains conversational context through an injected history store.
        """
        chat_history = self.history_store.get_messages(session_id, limit=self.max_history_messages)

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
                top_k_scenarios=5,
                top_k_advice=5,
                top_n_final=3
            )
            context = self.rag_pipeline.format_retrieved_context(search_output)

            # 7. Generate Response using LLM
            english_response = self._generate_final_response(english_message, emotion, context, chat_history)
            
        else:
            english_response = self._get_static_response("out_of_scope")

        # 8. Update and Trim Chat History
        self.history_store.append_turn(
            session_id=session_id,
            user_message=english_message,
            assistant_message=english_response,
            limit=self.max_history_messages,
        )

        # 9. Translate back if necessary
        if language_code not in ["en", "unknown"]:
            final_response = self._translate_from_english(english_response, language_name)
        else:
            final_response = english_response

        return final_response
