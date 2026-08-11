"""
FastAPI backend exposing the chatbot as a REST API.
This is the "production interface" — the same engine could be called by
a website widget, a mobile app, or the Streamlit demo below.

Run with:
    uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chatbot_engine import ChatbotEngine

app = FastAPI(
    title="Intelligent Customer Support Chatbot API",
    description="Hybrid intent-classification + generative-fallback chatbot.",
    version="1.0.0",
)

# Allow the Streamlit demo (or any frontend) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The engine loads the trained model once at startup and is reused for
# every request — loading it per-request would be far too slow.
engine: ChatbotEngine | None = None


@app.on_event("startup")
def load_engine() -> None:
    global engine
    print("Loading chatbot engine (models + embeddings)...")
    engine = ChatbotEngine()
    print("Chatbot engine ready.")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    confidence: float
    response: str
    used_fallback: bool
    escalate_to_human: bool


@app.get("/health")
def health_check():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    result = engine.process_message(request.session_id, request.message)
    return result


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")
    engine.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
