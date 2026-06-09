from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import BASE_DIR, EMOTION_MODEL_DIR, LANGUAGE_MODEL_DIR, VOCAB_DICT_DIR
from app.services.chat_history_store import InMemoryChatHistoryStore, SQLiteChatHistoryStore
from app.services.chatbot_orchestrator import ChatbotOrchestrator
from app.services.emotion_classifier import EmotionClassifier
from app.services.intent_classifier import IntentClassifier
from app.services.language_detector import LanguageDetector
from app.services.speech_to_text import SpeechToText
from app.services.rag import RAGPipeline

def build_orchestrator(use_memory_store: bool = False, max_history_turns: int = 5) -> ChatbotOrchestrator:
    history_store = InMemoryChatHistoryStore()
    if not use_memory_store:
        history_db_path = BASE_DIR / "data" / "chat_history.sqlite3"
        history_store = SQLiteChatHistoryStore(str(history_db_path))

    return ChatbotOrchestrator(
        intent_classifier=IntentClassifier(),
        emotion_classifier=EmotionClassifier(model_path=EMOTION_MODEL_DIR, vocab_path=VOCAB_DICT_DIR),
        language_detector=LanguageDetector(model_path=LANGUAGE_MODEL_DIR),
        stt_service=SpeechToText(),
        rag_pipeline=RAGPipeline(),
        history_store=history_store,
        max_history_turns=max_history_turns,
    )

def interactive_chat(orchestrator: ChatbotOrchestrator, session_id: str) -> None:
    print("Mental Health Support Chatbot")
    print("Type '/exit' to quit. Press Ctrl+C to stop.")
    print(f"Session ID: {session_id}")
    print("-" * 60)

    while True:
        try:
            user_message = input("You: ").strip()
        except EOFError:
            print("\nExiting chat.")
            break
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

        if not user_message:
            continue

        if user_message.lower() in {"/exit", "exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            response = orchestrator.process_message(user_message=user_message, session_id=session_id)
            print(f"Bot: {response}")
        except Exception as exc:
            print(f"Bot error: {exc}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mental health support chatbot.")
    parser.add_argument("--session-id", default="default", help="Conversation session identifier.")
    parser.add_argument(
        "--message",
        help="Send a single message to the chatbot and exit instead of starting interactive mode.",
    )
    parser.add_argument(
        "--audio",
        help="Path to an audio file to process through Whisper STT and exit.",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use in-memory chat history instead of SQLite persistence.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=5,
        help="Number of recent turns to keep in the prompt history.",
    )
    args = parser.parse_args()

    orchestrator = build_orchestrator(use_memory_store=args.memory, max_history_turns=args.history_turns)

    # 1. Handle Audio Override First
    if args.audio:
        print(f"Processing audio file: {args.audio} ...")
        try:
            response = orchestrator.process_message(audio_file_path=args.audio, session_id=args.session_id)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Failed to process audio: {e}")
        return

    # 2. Handle Single Text Message
    if args.message:
        response = orchestrator.process_message(user_message=args.message, session_id=args.session_id)
        print(response)
        return

    # 3. Handle Standard Interactive Loop
    interactive_chat(orchestrator, session_id=args.session_id)

if __name__ == "__main__":
    main()