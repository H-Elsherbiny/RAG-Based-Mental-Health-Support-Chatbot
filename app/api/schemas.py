from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to send to the chatbot")
    session_id: str = Field(default="default", description="Conversation session identifier")


class ChatResponse(BaseModel):
    session_id: str
    input_message: str
    response: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
