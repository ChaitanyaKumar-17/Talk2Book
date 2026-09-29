"""Behavior checks for the deterministic calendar tools."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.llm import VoiceAssistant
from backend import main
from backend.mock_calendar import CalendarStore
from backend.stt import DeepgramStream
from backend.tools import book_appointment, check_availability


class CalendarToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.calendar = CalendarStore(Path(self.temp_directory.name) / "calendar.sqlite3")
        self.appointment_date = (date.today() + timedelta(days=2)).isoformat()

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_check_availability_returns_half_hour_slots_in_requested_window(self) -> None:
        result = check_availability(self.calendar, self.appointment_date, "afternoon")

        self.assertEqual(result["date"], self.appointment_date)
        self.assertEqual(result["available_slots"][0], "12:00")
        self.assertEqual(result["available_slots"][-1], "16:30")
        self.assertTrue(all(slot >= "12:00" for slot in result["available_slots"]))

    def test_booking_removes_slot_from_availability(self) -> None:
        booking = book_appointment(self.calendar, "Ada Lovelace", self.appointment_date, "14:00")

        self.assertTrue(booking["success"])
        result = check_availability(self.calendar, self.appointment_date, "afternoon")
        self.assertNotIn("14:00", result["available_slots"])

    def test_double_booking_returns_conflict_without_overwriting_first_booking(self) -> None:
        first = book_appointment(self.calendar, "Ada Lovelace", self.appointment_date, "10:30")
        second = book_appointment(self.calendar, "Grace Hopper", self.appointment_date, "10:30")

        self.assertTrue(first["success"])
        self.assertFalse(second["success"])
        self.assertEqual(self.calendar.list_appointments()[0]["name"], "Ada Lovelace")

    def test_rejects_invalid_date_and_out_of_hours_booking(self) -> None:
        with self.assertRaises(ValueError):
            check_availability(self.calendar, "next Tuesday", "morning")
        with self.assertRaises(ValueError):
            book_appointment(self.calendar, "Ada", self.appointment_date, "17:00")


class StreamingResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_sentence_is_synthesized_before_llm_stream_finishes(self) -> None:
        spoken_sentences: list[str] = []

        def chunk(content: str):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))]
            )

        class FakeStream:
            async def __aiter__(self):
                yield chunk("First sentence. ")
                await asyncio.sleep(0.01)
                if not spoken_sentences:
                    raise AssertionError("sentence synthesis waited for the LLM stream to finish")
                yield chunk("Second sentence.")

        class FakeCompletions:
            async def create(self, **kwargs):
                return FakeStream()

        assistant = object.__new__(VoiceAssistant)
        assistant.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        assistant.model = "test-model"

        async def record_sentence(sentence: str) -> None:
            spoken_sentences.append(sentence)

        async def ignore_tool(name: str, result: dict[str, object]) -> None:
            return None

        response = await assistant.respond(
            [{"role": "user", "content": "Say two sentences."}],
            record_sentence,
            ignore_tool,
            lambda: None,
        )

        self.assertEqual(response, "First sentence. Second sentence.")
        self.assertEqual(spoken_sentences, ["First sentence.", "Second sentence."])


class DeepgramAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_boolean_sdk_methods_are_run_without_awaiting_results(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeConnection:
            def on(self, event, callback) -> None:
                calls.append(("on", event))

            def start(self, options) -> bool:
                calls.append(("start", options))
                return True

            def send(self, audio: bytes) -> bool:
                calls.append(("send", audio))
                return True

            def keep_alive(self) -> bool:
                calls.append(("keep_alive", True))
                return True

            def finish(self) -> bool:
                calls.append(("finish", True))
                return True

            def finalize(self) -> bool:
                calls.append(("finalize", True))
                return True

        connection = FakeConnection()
        fake_client = SimpleNamespace(
            listen=SimpleNamespace(websocket=SimpleNamespace(v=lambda version: connection))
        )
        fake_events = SimpleNamespace(Transcript="transcript", UtteranceEnd="utterance_end")
        loop = asyncio.get_running_loop()
        stream = DeepgramStream("test-key", AsyncMock(), AsyncMock(), loop)

        with (
            patch("deepgram.DeepgramClient", return_value=fake_client),
            patch("deepgram.LiveOptions", return_value="test-options"),
            patch("deepgram.LiveTranscriptionEvents", fake_events),
        ):
            await stream.start()
            await stream.send(b"pcm")
            self.assertTrue(await stream.keep_alive())
            await stream.finalize()
            await stream.close()

        self.assertEqual(
            [name for name, _ in calls],
            ["on", "on", "start", "send", "keep_alive", "finalize", "finish"],
        )
        self.assertTrue(stream._manual_finalize_pending)
        self.assertIn(("send", b"pcm"), calls)

    async def test_manual_finalize_uses_buffered_text_from_empty_finalize_result(self) -> None:
        stream = DeepgramStream("test-key", AsyncMock(), AsyncMock(), asyncio.get_running_loop())
        stream._manual_finalize_pending = True
        stream._final_parts.append("What is open tomorrow afternoon?")
        result = SimpleNamespace(
            channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript="")]),
            is_final=True,
            speech_final=False,
            from_finalize=True,
        )

        with patch.object(stream, "_dispatch") as dispatch:
            stream._on_transcript(result=result)

        dispatch.assert_called_once_with(stream.on_final, "What is open tomorrow afternoon?")
        self.assertFalse(stream._manual_finalize_pending)

    async def test_transcript_callback_accepts_result_as_argument_and_keyword(self) -> None:
        stream = DeepgramStream("test-key", AsyncMock(), AsyncMock(), asyncio.get_running_loop())
        result = SimpleNamespace(
            channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript="hello")]),
            is_final=False,
            speech_final=False,
        )

        with patch.object(stream, "_dispatch") as dispatch:
            stream._on_transcript(result, result=result)

        dispatch.assert_called_once_with(stream.on_partial, "hello")


class VoiceConnectionErrorTests(unittest.TestCase):
    def test_startup_exception_is_reported_without_python_details(self) -> None:
        class FailingDeepgramStream:
            def __init__(self, *args) -> None:
                pass

            async def start(self) -> None:
                raise RuntimeError("object bool can't be used in await expression")

            async def close(self) -> None:
                pass

        with (
            patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test", "GROQ_API_KEY": "test"}),
            patch("backend.main.DeepgramStream", FailingDeepgramStream),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/ws") as websocket:
                message = websocket.receive_json()

        self.assertEqual(message["type"], "error")
        self.assertIn("internet connection", message["message"])
        self.assertNotIn("RuntimeError", message["message"])
        self.assertNotIn("await", message["message"])


class TranscriptSubmissionTests(unittest.TestCase):
    def test_final_transcript_waits_for_send_button_before_assistant_reply(self) -> None:
        instances = []

        class FakeDeepgramStream:
            def __init__(self, api_key, on_partial, on_final, loop) -> None:
                self.on_final = on_final
                self.loop = loop
                self.endpointing_ms = 1200
                instances.append(self)

            async def start(self) -> None:
                return None

            async def send(self, audio: bytes) -> None:
                return None

            async def finalize(self) -> None:
                await self.on_final("What is open tomorrow afternoon?")

            async def close(self) -> None:
                return None

        class FakeAssistant:
            calls = []

            def __init__(self, *args) -> None:
                pass

            async def respond(self, history, on_sentence, on_tool, on_first_token):
                self.calls.append(history[-1]["content"])
                on_first_token()
                await on_sentence("I found three open slots.")
                return "I found three open slots."

        class FakeSynthesizer:
            def synthesize(self, text: str) -> bytes:
                return b"test-wav"

        FakeAssistant.calls.clear()
        with (
            patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test", "GROQ_API_KEY": "test"}),
            patch("backend.main.DeepgramStream", FakeDeepgramStream),
            patch("backend.main.VoiceAssistant", FakeAssistant),
            patch("backend.main.synthesizer", FakeSynthesizer()),
            patch.object(main.latency, "finish_turn"),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/ws") as websocket:
                self.assertEqual(websocket.receive_json(), {"type": "config", "silence_timeout_ms": 1200})
                self.assertEqual(websocket.receive_json(), {"type": "status", "value": "listening"})
                websocket.send_json({"type": "finalize_transcript"})
                self.assertEqual(websocket.receive_json(), {"type": "status", "value": "finalizing"})
                self.assertEqual(
                    websocket.receive_json(),
                    {"type": "transcript", "text": "What is open tomorrow afternoon?", "final": True},
                )
                self.assertEqual(websocket.receive_json(), {"type": "status", "value": "ready_to_send"})
                self.assertEqual(FakeAssistant.calls, [])

                websocket.send_json(
                    {"type": "submit_transcript", "text": "What is open tomorrow afternoon?"}
                )
                self.assertEqual(websocket.receive_json(), {"type": "status", "value": "thinking"})
                self.assertEqual(websocket.receive_json()["type"], "audio")
                self.assertEqual(websocket.receive_json(), {"type": "turn_complete"})
                self.assertEqual(websocket.receive_json(), {"type": "status", "value": "listening"})

        self.assertEqual(FakeAssistant.calls, ["What is open tomorrow afternoon?"])


if __name__ == "__main__":
    unittest.main()