"""Groq streaming chat and function-calling orchestration."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from collections.abc import Awaitable, Callable
from datetime import date

from backend.mock_calendar import CalendarStore
from backend.tools import book_appointment, check_availability

SentenceCallback = Callable[[str], Awaitable[None]]
ToolCallback = Callable[[str, dict[str, object]], Awaitable[None]]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check open appointment slots for a date and part of day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time_range": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "any"],
                    },
                },
                "required": ["date", "time_range"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Reserve one available half-hour appointment slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "Start time in HH:MM 24-hour format."},
                },
                "required": ["name", "date", "time"],
            },
        },
    },
]


class VoiceAssistant:
    def __init__(self, api_key: str, calendar: CalendarStore, model: str = "openai/gpt-oss-120b") -> None:
        from groq import AsyncGroq

        self.client = AsyncGroq(api_key=api_key)
        self.calendar = calendar
        self.model = model

    async def respond(
        self,
        history: list[dict[str, object]],
        on_sentence: SentenceCallback,
        on_tool: ToolCallback,
        on_first_token: Callable[[], None],
    ) -> str:
        user_turns = [index for index, message in enumerate(history) if message.get("role") == "user"]
        if len(user_turns) > 6:
            history[:] = history[user_turns[-6] :]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise, warm appointment receptionist. Today is "
                    f"{date.today().isoformat()}. Use the calendar tools for availability and bookings; "
                    "never claim a booking succeeded unless the tool confirms it. Clarify ambiguous dates, "
                    "names, and times. Business hours are 09:00-17:00 in half-hour slots. "
                    "Speak in natural short sentences suitable for audio."
                ),
            },
            *history[-12:],
        ]
        visible_response = ""
        first_token_seen = False

        for _ in range(3):
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                stream=True,
            )
            sentence_buffer = ""
            tool_calls: dict[int, dict[str, str]] = {}
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def synthesize_queued_sentences() -> None:
                while sentence := await sentence_queue.get():
                    await on_sentence(sentence)

            speech_worker = asyncio.create_task(synthesize_queued_sentences())

            try:
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.tool_calls and not first_token_seen:
                        first_token_seen = True
                        on_first_token()
                    if delta.content:
                        if not first_token_seen:
                            first_token_seen = True
                            on_first_token()
                        visible_response += delta.content
                        sentence_buffer += delta.content
                        while match := re.search(r"[.!?](?:[\"')\]]*)\s+", sentence_buffer):
                            sentence = sentence_buffer[: match.end()].strip()
                            sentence_buffer = sentence_buffer[match.end() :]
                            if sentence:
                                await sentence_queue.put(sentence)
                    for tool_delta in delta.tool_calls or []:
                        call = tool_calls.setdefault(
                            tool_delta.index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if tool_delta.id:
                            call["id"] = tool_delta.id
                        if tool_delta.function and tool_delta.function.name:
                            call["name"] += tool_delta.function.name
                        if tool_delta.function and tool_delta.function.arguments:
                            call["arguments"] += tool_delta.function.arguments

                if sentence_buffer.strip():
                    await sentence_queue.put(sentence_buffer.strip())
                await sentence_queue.put(None)
                await speech_worker
            except BaseException:
                speech_worker.cancel()
                with suppress(asyncio.CancelledError):
                    await speech_worker
                raise

            assistant_tool_calls = []
            for index in sorted(tool_calls):
                call = tool_calls[index]
                assistant_tool_calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                )

            if assistant_tool_calls:
                messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})
                for call in assistant_tool_calls:
                    name = call["function"]["name"]
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                        result = self._run_tool(name, arguments)
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                        result = {"success": False, "error": str(error)}
                    await on_tool(name, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result),
                        }
                    )
                continue

            break

        history[:] = messages[1:]
        return visible_response

    def _run_tool(self, name: str, arguments: dict[str, str]) -> dict[str, object]:
        if name == "check_availability":
            return check_availability(self.calendar, **arguments)
        if name == "book_appointment":
            return book_appointment(self.calendar, **arguments)
        raise ValueError(f"Unknown calendar tool: {name}")
