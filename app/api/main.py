from fastapi import FastAPI, HTTPException, Request

from app.main import build_orchestrator
from app.api.schemas import ChatHistoryResponse, ChatRequest, ChatResponse, ChatMessage


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mental Health Support Chatbot API",
        version="1.0.0",
        description="HTTP endpoints for interacting with the chatbot orchestrator.",
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.orchestrator = build_orchestrator(use_memory_store=False)
        app.state.history_store = app.state.orchestrator.history_store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "mental-health-support-chatbot"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, app_request: Request) -> ChatResponse:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="message cannot be empty")

        orchestrator = app_request.app.state.orchestrator
        response_text = orchestrator.process_message(request.message, session_id=request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            input_message=request.message,
            response=response_text,
        )

    @app.get("/history/{session_id}", response_model=ChatHistoryResponse)
    def get_history(session_id: str, app_request: Request) -> ChatHistoryResponse:
        history_store = app_request.app.state.history_store
        messages = history_store.get_messages(session_id)
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[ChatMessage(**message) for message in messages],
        )

    return app


app = create_app()
