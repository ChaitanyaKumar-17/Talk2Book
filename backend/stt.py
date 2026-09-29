"""Deepgram live transcription adapter for 16 kHz mono PCM audio."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable


class DeepgramStream:
    def __init__(
        self,
        api_key: str,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.api_key = api_key
        self.on_partial = on_partial
        self.on_final = on_final
        self.loop = loop
        self.connection = None
        self._final_parts: list[str] = []
        self.endpointing_ms = 1200
        self._manual_finalize_pending = False
        self._write_lock: asyncio.Lock | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._last_activity = 0.0

    async def start(self) -> None:
        from deepgram import (
            DeepgramClient,
            LiveOptions,
            LiveTranscriptionEvents,
        )

        try:
            endpointing_ms = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "1200"))
        except ValueError:
            endpointing_ms = 1200
        endpointing_ms = min(5000, max(1000, endpointing_ms))
        self.endpointing_ms = endpointing_ms

        client = DeepgramClient(self.api_key)
        self.connection = client.listen.websocket.v("1")
        self._write_lock = asyncio.Lock()
        self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms=str(endpointing_ms),
            endpointing=endpointing_ms,
            vad_events=True,
        )
        started = await asyncio.to_thread(self.connection.start, options)
        if not started:
            raise ConnectionError("Deepgram could not start the transcription stream")
        self._last_activity = self.loop.time()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _dispatch(self, callback: Callable[[str], Awaitable[None]], text: str) -> None:
        asyncio.run_coroutine_threadsafe(callback(text), self.loop)

    def _on_transcript(self, *args, **kwargs) -> None:
        result = kwargs.get("result")
        if result is None and args:
            result = args[0]
        try:
            alternative = result.channel.alternatives[0]
            transcript = alternative.transcript.strip()
        except (AttributeError, IndexError, TypeError):
            return
        is_final = getattr(result, "is_final", False)
        should_finalize = (
            getattr(result, "speech_final", False)
            or getattr(result, "from_finalize", False)
            or (self._manual_finalize_pending and is_final)
        )
        if transcript:
            if is_final:
                self._final_parts.append(transcript)
            else:
                self._dispatch(self.on_partial, transcript)
        if should_finalize:
            self._dispatch_final(transcript)

    def _dispatch_final(self, fallback: str = "") -> None:
        final_text = " ".join(self._final_parts).strip() or fallback
        self._final_parts.clear()
        if final_text:
            self._manual_finalize_pending = False
            self._dispatch(self.on_final, final_text)

    def _on_utterance_end(self, *args, **kwargs) -> None:
        self._dispatch_final()

    async def send(self, audio: bytes) -> None:
        if self.connection is not None and self._write_lock is not None:
            async with self._write_lock:
                sent = await asyncio.to_thread(self.connection.send, audio)
            if not sent:
                raise ConnectionError("Deepgram could not accept the audio frame")
            self._last_activity = asyncio.get_running_loop().time()

    async def keep_alive(self) -> bool:
        if self.connection is None or self._write_lock is None:
            return False
        async with self._write_lock:
            sent = await asyncio.to_thread(self.connection.keep_alive)
        if sent:
            self._last_activity = asyncio.get_running_loop().time()
        return sent

    async def _keepalive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                await asyncio.sleep(3)
                if loop.time() - self._last_activity >= 3 and not await self.keep_alive():
                    return
        except asyncio.CancelledError:
            raise

    async def finalize(self) -> None:
        if self.connection is not None and self._write_lock is not None:
            self._manual_finalize_pending = True
            async with self._write_lock:
                finalized = await asyncio.to_thread(self.connection.finalize)
            if not finalized:
                self._manual_finalize_pending = False
                raise ConnectionError("Deepgram could not finalize the current transcript")
            self._last_activity = asyncio.get_running_loop().time()

    async def close(self) -> None:
        if self.connection is not None:
            if self._keepalive_task is not None:
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass
                self._keepalive_task = None
            if self._write_lock is not None:
                async with self._write_lock:
                    await asyncio.to_thread(self.connection.finish)
