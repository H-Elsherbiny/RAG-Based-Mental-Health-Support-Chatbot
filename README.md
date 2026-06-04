# RAG-Based Mental Health Support Chatbot

An AI-driven mental health support chatbot that leverages Retrieval-Augmented Generation (RAG) to provide empathetic, contextually aware, and safe responses. By combining multiple NLP classification pipelines, two-stage hybrid search, and cross-encoder reranking, the system retrieves relevant historical counselor advice from a vector database and synthesizes actionable support using a Large Language Model, all while maintaining rigorous conversational safety.

## 2. Technical Architecture & Tech Stack

The system is built with a modular, microservice-inspired architecture, emphasizing distinct NLP tasks (language detection, intent classification, emotion recognition) before executing the RAG pipeline and final response generation.

### Tech Stack

- **Interface**: Gradio (for interactive interface) & FastAPI (for robust, production-ready REST endpoints).
- **Core Logic & Orchestration**: Python 3.10+, orchestrated via custom service classes.
- **Natural Language Processing (NLP)**:
  - **Language Detection**: Custom Hierarchical pipeline using Scikit-Learn/Joblib & FastText regex patterns.
  - **Emotion Classification**: Custom PyTorch BiLSTM model using spaCy for tokenization.
  - **Intent Classification**: Few-shot prompted LLM via Groq API (`openai/gpt-oss-20b`).
- **Database/Vector Store**: Qdrant (Cloud/Remote) for storing and querying dense and sparse embeddings.
- **Search & Embeddings**:
  - Dense Embeddings: `fastembed` (`BAAI/bge-base-en-v1.5`)
  - Sparse Embeddings: `fastembed` (`prithivida/Splade_PP_en_v1`)
  - Reranking: `sentence-transformers` CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM/Generation**: Groq API (`openai/gpt-oss-20b`) for high-speed inference (translation, query rewriting, response synthesis).
- **MLOps/Telemetry**: Langfuse for prompt observability, generation tracing, and performance monitoring.
- **State Management**: SQLite (persistent) & In-Memory chat history buffering.

### Architecture Flow

1. **Input Processing**: The `ChatbotOrchestrator` receives a message, logs it in the `ChatHistoryStore`, and runs `LanguageDetector` to identify the script/language. Non-English queries are translated.
2. **Classification**: The `IntentClassifier` determines if the query requires mental health support, greeting, or is out of scope. The `EmotionClassifier` extracts the user's emotional state using a BiLSTM model.
3. **Retrieval**: The `RAGPipeline` rewrites the user query for maximum vector retrieval, performs a Two-Stage Hybrid Search (Dense + Sparse) on Qdrant, and filters counselor advice based on similar patient scenarios. The results are reranked via a CrossEncoder.
4. **Generation**: The LLM synthesizes a final empathetic response incorporating the retrieved advice, user emotion, and chat history. The response is translated back to the original language if necessary.

## 3. Key Features

- **Advanced Query Rewriting**: Utilizes a fast Groq model to strip conversational noise and output semantically dense search strings optimized for vector database lookups.
- **Two-Stage Hybrid Search & Reranking**: Implements dense (`BAAI/bge-base-en-v1.5`) and sparse (`prithivida/Splade_PP_en_v1`) embeddings for comprehensive retrieval, followed by deep attention-based cross-encoder reranking to ensure absolute contextual relevance.
- **Multilingual Support**: Integrates an automatic language detection module and translates non-English queries for internal processing, returning the final response in the user's native language.
- **Emotion & Intent Classification**: Employs a custom PyTorch BiLSTM for emotion detection and a few-shot LLM approach for intent classification, ensuring the chatbot responds appropriately to greetings versus actual distress calls.
- **Contextual Memory Buffering**: Maintains a configurable sliding window of chat history (via SQLite or In-Memory) to provide conversational continuity without exceeding LLM context limits.
- **Safety & Guardrails**: Hardcoded system prompts prevent the AI from diagnosing conditions, recommending medications, or breaking character, gracefully handling out-of-scope requests.
- **Telemetry & Tracing**: fully integrated with Langfuse (`@observe` decorators) to monitor latency, track LLM costs, and debug pipeline steps in real-time.

## 4. Prerequisites

Before setting up the project, ensure you have the following installed and configured:

- **Python**: Version 3.10 or higher.
- **API Accounts**:
  - [Groq API Key](https://console.groq.com/) for LLM access.
  - [Qdrant Database](https://qdrant.tech/) URL and API Key for vector storage.
  - [Langfuse](https://langfuse.com/) Secret/Public keys for telemetry.

## 5. Installation & Local Setup

Run the following commands in your terminal to set up the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/H-Elsherbiny/RAG-Based-Mental-Health-Support-Chatbot.git
cd RAG-Based-Mental-Health-Support-Chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

### Environment Variables

Create a `.env` file in the root directory and populate it with your credentials. **Never commit this file to version control.**

| Variable Name | Purpose | Where to get it |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Authenticates LLM inference requests. | Groq Console Dashboard |
| `QDRANT_URL` | Endpoint for your Qdrant Vector DB cluster. | Qdrant Cloud Dashboard |
| `QDRANT_API_KEY` | Authenticates access to the Qdrant DB. | Qdrant Cloud Dashboard |
| `LANGFUSE_SECRET_KEY` | Telemetry backend secret key. | Langfuse Project Settings |
| `LANGFUSE_PUBLIC_KEY` | Telemetry backend public key. | Langfuse Project Settings |
| `LANGFUSE_BASE_URL` | Base URL for Langfuse instance. | Langfuse Project Settings |

## 6. Usage & Running the Application

The project supports three modes of execution: a Gradio Web Interface, a FastAPI backend, and a Command-Line Interface (CLI).

### Gradio Interface (Interactive UI)
To launch the user-friendly web interface:
```bash
python gradio_app.py
```
*Navigate to `http://localhost:7860` in your browser to interact with the chatbot.*

### FastAPI Server (REST API)
To start the production-ready API server:
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```
*Navigate to `http://localhost:8000/docs` to view the interactive Swagger UI and test the `/chat` and `/history/{session_id}` endpoints.*

### Command-Line Interface (CLI)
To run the chatbot interactively in your terminal:
```bash
# Start an interactive chat session
python app/main.py

# Send a single message and exit
python app/main.py --message "I have a panic attack because I have an important presentation tomorrow. I am afraid that I did not prepare well."
```

## 7. Project Directory Structure

```text
RAG-Based-Mental-Health-Support-Chatbot/
├── app/                        # Main application source code
│   ├── api/                    # FastAPI endpoints and schemas
│   ├── core/                   # Global configuration and environment loading
│   ├── services/               # Core NLP, classification, and RAG pipelines
│   │   ├── emotion_classifier/ # PyTorch BiLSTM emotion inference
│   │   ├── intent_classifier/  # Groq-based few-shot intent routing
│   │   ├── language_detector/  # Custom hierarchical language detection
│   │   ├── rag/                # Two-stage Qdrant hybrid search & cross-encoder
│   │   ├── chat_history_store.py # SQLite and In-Memory state management
│   │   └── chatbot_orchestrator.py # Pipeline coordinator (The "Brain")
│   └── main.py                 # CLI entry point
├── models/                     # Serialized machine learning models and vocabularies
├── notebooks/                  # Research, training, and evaluation Jupyter notebooks
├── gradio_app.py               # Gradio web interface entry point
├── requirements.txt            # Python dependencies
└── .env.example                # Template for environment variables
```

---
