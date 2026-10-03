"""FastAPI WebSocket server for the Talk2Book voice receptionist."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.latency import LatencyRecorder
from backend.llm import VoiceAssistant
from backend.mock_calendar import CalendarStore
from backend.stt import DeepgramStream
from backend.tts import EdgeTtsSynthesizer

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
load_dotenv(ROOT / ".env")

app = FastAPI(title="Talk2Book Voice Receptionist", version="1.0.0")
calendar = CalendarStore(os.getenv("CALENDAR_DB_PATH", str(ROOT / "data" / "calendar.sqlite3")))
latency = LatencyRecorder(ROOT / "results" / "latency_report.json")
synthesizer = EdgeTtsSynthesizer(
    voice_name=os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural"),
    rate=os.getenv("EDGE_TTS_RATE", "+8%"),
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "talk2book"})


@app.get("/api/appointments")
async def appointments() -> list[dict[str, object]]:
    return calendar.list_appointments()


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not api_key or not groq_key:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Set DEEPGRAM_API_KEY and GROQ_API_KEY in .env before starting a voice session.",
            }
        )
        await websocket.close(code=1011)
        return

    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    turn_task: asyncio.Task | None = None
    history: list[dict[str, object]] = []
    assistant = VoiceAssistant(groq_key, calendar, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    async def send(payload: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def handle_utterance(transcript: str) -> None:
        nonlocal turn_task
        current_task = asyncio.current_task()
        if turn_task is not None and turn_task is not current_task and not turn_task.done():
            turn_task.cancel()
        turn_task = current_task
        timing = latency.start_turn()
        history.append({"role": "user", "content": transcript})
        await send({"type": "status", "value": "thinking"})
        llm_started_at = time.perf_counter()
        first_token_at: float | None = None
        first_audio_at: float | None = None
        tts_seconds = 0.0

        def mark_first_token() -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.perf_counter()

        async def speak(sentence: str) -> None:
            nonlocal first_audio_at, tts_seconds
            synthesis_started = time.perf_counter()
            audio_bytes = await asyncio.to_thread(synthesizer.synthesize, sentence)
            tts_seconds += time.perf_counter() - synthesis_started
            if not audio_bytes:
                return
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            await send(
                {
                    "type": "audio",
                    "encoding": "audio/mpeg",
                    "audio": base64.b64encode(audio_bytes).decode("ascii"),
                    "text": sentence,
                }
            )

        async def report_tool(name: str, result: dict[str, object]) -> None:
            await send({"type": "tool_result", "name": name, "result": result})

        async def report_summary(summary: dict[str, object]) -> None:
            await send({"type": "order_summary", "summary": summary})

        try:
            response = await assistant.respond(history, speak, report_tool, mark_first_token, report_summary)
            history.append({"role": "assistant", "content": response or ""})
            await send({"type": "turn_complete"})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Assistant turn failed")
            await send(
                {
                    "type": "error",
                    "message": "I couldn't finish that response. Please try again. If it keeps happening, check the server connection and API settings.",
                }
            )
        finally:
            latency.finish_turn(
                timing,
                llm_started_at=llm_started_at,
                first_token_at=first_token_at,
                first_audio_at=first_audio_at,
                tts_seconds=tts_seconds,
            )
            if websocket.client_state.name != "DISCONNECTED":
                with suppress(Exception):
                    await send({"type": "status", "value": "listening"})

    async def partial_update(text: str) -> None:
        await send({"type": "transcript", "text": text, "final": False})

    async def transcript_ready(text: str) -> None:
        await send({"type": "transcript", "text": text, "final": True})
        await send({"type": "status", "value": "ready_to_send"})

    stt = DeepgramStream(api_key, partial_update, transcript_ready, loop)
    try:
        await stt.start()
        await send({"type": "config", "silence_timeout_ms": stt.endpointing_ms})
        await send({"type": "status", "value": "listening"})
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                await stt.send(message["bytes"])
            elif message.get("text"):
                import json

                control = json.loads(message["text"])
                if control.get("type") == "barge_in" and turn_task and not turn_task.done():
                    turn_task.cancel()
                    await send({"type": "status", "value": "listening"})
                elif control.get("type") == "finalize_transcript":
                    await send({"type": "status", "value": "finalizing"})
                    await stt.finalize()
                elif control.get("type") == "submit_transcript":
                    transcript = control.get("text")
                    if isinstance(transcript, str) and transcript.strip():
                        if turn_task is None or turn_task.done():
                            turn_task = asyncio.create_task(handle_utterance(transcript.strip()))
                    else:
                        await send({"type": "error", "message": "I didn't catch a complete phrase. Please speak, wait for the transcript, then press Send."})
                elif control.get("type") == "ping":
                    await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        if websocket.client_state.name != "DISCONNECTED":
            logger.exception("Voice connection failed")
            with suppress(Exception):
                await send(
                    {
                        "type": "error",
                        "message": "The voice service couldn't connect. Check your internet connection and API settings, then try again.",
                    }
                )
    finally:
        if turn_task and not turn_task.done():
            turn_task.cancel()
            with suppress(asyncio.CancelledError):
                await turn_task
        with suppress(Exception):
            await stt.close()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
