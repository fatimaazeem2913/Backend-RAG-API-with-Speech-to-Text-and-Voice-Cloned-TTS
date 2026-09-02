import json
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag_service import EnterpriseRAGService
from src.stt_service import STTService
from src.tts_service import TTSService
from src.pipeline_log import Timeline, configure as configure_pipeline_log

# 🚀 Persistent logging: writes to backend/logs/app.log (rotates at 5MB,
# keeps 3 backups) AND still prints to the terminal like before.
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
configure_pipeline_log(LOG_DIR)  # sets up the [request_id] event t+/delta trace lines

logger = logging.getLogger("voice_rag")
logger.setLevel(logging.INFO)

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_formatter)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

if not logger.handlers:
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

app = FastAPI(title="Enterprise Voice RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = EnterpriseRAGService()
stt_service = STTService()
tts_service = TTSService()

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    message: str
    strategy: str = "hybrid"

class TTSRequest(BaseModel):
    text: str
    session_id: str = "default_session"
    request_id: Optional[str] = None  # ties this TTS call's Timeline to the chat call's, for correlated logs

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("[STT WS] Live raw audio stream connected.")

    SR = 16000
    WINDOW_SEC = 6                      # bounded window -> Whisper call time stays ~flat
    THROTTLE_SAMPLES = int(SR * 0.35)   # ~350ms between partial updates
    SILENCE_COMMIT_SAMPLES = int(SR * 0.7)  # 700ms of quiet finalizes the current phrase

    window = np.array([], dtype=np.float32)
    committed_text = ""
    last_processed_len = 0
    silence_run = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            window = np.concatenate((window, chunk))

            chunk_rms = np.sqrt(np.mean(np.square(chunk))) if len(chunk) else 0.0
            silence_run = 0 if chunk_rms > 0.01 else silence_run + len(chunk)

            # 🚀 THE FIX: never let the transcription window grow unbounded.
            # Re-transcribing a 20s+ buffer every throttle tick is what made
            # live text appear to "freeze" until you stopped talking.
            max_len = SR * WINDOW_SEC
            if len(window) > max_len:
                trim = len(window) - max_len
                window = window[-max_len:]
                last_processed_len = max(0, last_processed_len - trim)

            # Pause detected -> finalize this phrase into committed_text and
            # start a fresh window, so long utterances don't get truncated.
            if silence_run >= SILENCE_COMMIT_SAMPLES and len(window) > 0:
                text = stt_service.transcribe_stream(window)
                if text:
                    committed_text = (committed_text + " " + text).strip()
                    await websocket.send_text(json.dumps({"text": committed_text, "final": True}))
                window = np.array([], dtype=np.float32)
                last_processed_len = 0
                silence_run = 0
                continue

            # Regular throttled partial update on the current (bounded) window
            if len(window) - last_processed_len >= THROTTLE_SAMPLES:
                partial = stt_service.transcribe_stream(window)
                last_processed_len = len(window)
                if partial:
                    live_text = (committed_text + " " + partial).strip()
                    logger.info(f"[STT WS Live]: {live_text}")
                    await websocket.send_text(json.dumps({"text": live_text, "final": False}))

    except WebSocketDisconnect:
        logger.info("[STT WS] Stream disconnected.")
    except Exception as e:
        logger.error(f"[STT WS Error] Backend pipeline issue: {e}", exc_info=True)
        
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Fallback endpoint for file upload transcription."""
    start_t = time.time()
    contents = await file.read()
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    text = stt_service.transcribe(contents, suffix=ext)
    elapsed = time.time() - start_t
    logger.info(f"[TRANSCRIBE] bytes={len(contents)} latency={elapsed:.2f}s result={text!r}")
    return {"text": text}

@app.post("/api/transcribe/stream-chunk")
async def transcribe_stream_chunk(file: UploadFile = File(...)):
    """Incremental chunk endpoint for live streaming transcription."""
    contents = await file.read()
    text = stt_service.transcribe(contents)
    return {"text": text}

@app.post("/api/rag/chat")
async def chat_endpoint(req: ChatRequest):
    """Non-streaming RAG chat — this is what the frontend's Send button and
    voice flow call. (The /stream variant below is for token-by-token SSE.)"""
    tl = Timeline("chat")
    tl.mark("chat.request_received", method=req.strategy, message_chars=len(req.message))
    try:
        result = rag_service.chat(
            session_id=req.session_id, message=req.message, strategy=req.strategy, timeline=tl
        )
        tl.mark("chat.response_sent", total_s=round(tl.elapsed(), 3))
        result["request_id"] = tl.request_id
        return result
    except Exception as e:
        logger.error(f"[CHAT ERROR] request_id={tl.request_id} error={e}", exc_info=True)
        return {
            "answer": "Sorry, something went wrong retrieving that answer. Please try again.",
            "citations": [],
            "request_id": tl.request_id,
        }

@app.post("/api/rag/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Token-by-token SSE streaming of RAG answers."""
    tl = Timeline("chat")
    tl.mark("chat.request_received", method=req.strategy, message_chars=len(req.message))

    def wrapped():
        for event in rag_service.chat_stream(req.session_id, req.message, req.strategy, timeline=tl):
            yield event
        tl.mark("chat.response_sent", total_s=round(tl.elapsed(), 3))

    return StreamingResponse(wrapped(), media_type="text/event-stream")

@app.post("/api/tts/stream")
async def stream_voice(req: TTSRequest):
    """Chunked raw 24kHz PCM stream of synthesized cloned voice."""
    tl = Timeline("tts_stream", request_id=req.request_id)
    tl.mark("tts_stream.request_received", text_chars=len(req.text), language="en")

    def wrapped():
        frame_count = 0
        wire_bytes = 0
        first_frame = True
        for chunk in tts_service.stream_audio(req.text, session_id=req.session_id, timeline=tl):
            frame_count += 1
            wire_bytes += len(chunk)
            if first_frame:
                first_frame = False
                tl.mark("tts_stream.first_frame_sent", bytes=wire_bytes)
            yield chunk
        tl.mark("tts_stream.last_frame_sent", frames=frame_count, wire_bytes=wire_bytes, total_s=round(tl.elapsed(), 3))

    return StreamingResponse(wrapped(), media_type="audio/pcm")
@app.on_event("shutdown")
def force_shutdown():
    logger.info("🛑 Forcefully flushing ML threads and shutting down...")
    os._exit(0)