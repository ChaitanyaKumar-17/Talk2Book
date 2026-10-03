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
            "description": "Check open one-hour turf booking slots and exact prices for a date and session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time_range": {
                        "type": "string",
                        "enum": ["morning", "evening", "any"],
                        "description": "Morning is 06:00-11:00; evening is 19:00-00:00; any returns both sessions.",
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
            "description": "Reserve one available one-hour turf slot and return its order ID and exact charge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "Available slot start time in HH:MM 24-hour format; duration is one hour."},
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
        on_summary: SummaryCallback | None = None,
    ) -> str:
        user_turns = [index for index, message in enumerate(history) if message.get("role") == "user"]
        if len(user_turns) > 6:
            history[:] = history[user_turns[-6] :]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise, friendly turf booking assistant for Northside Turf. Today is "
                    f"{date.today().isoformat()}. Use the calendar tools for availability and bookings; "
                    "never claim a booking succeeded unless the tool confirms it. Clarify ambiguous dates, "
                    "names, and times. There is one field with one-hour slots. Morning session is 06:00-11:00; "
                    "evening session is 19:00-00:00. Do not offer afternoon availability. "
                    "Weekday morning slots cost 1500 rupees; weekday evening slots cost 2500 rupees. "
                    "Weekend morning and evening slots cost 3000 rupees. Quote charges only from tool results. "
                    "After a successful booking, give a compact receipt on separate lines with exactly: "
                    "Order ID, Name, Time, Charges. Then remind the player to arrive at the physical counter "
                    "10 minutes before the booking and pay there; if they do not pay, the slot may be assigned "
                    "to someone else. Do not say payment has been received; this system takes no payment. "
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
                confirmed_booking: dict[str, object] | None = None
                for call in assistant_tool_calls:
                    name = call["function"]["name"]
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                        result = self._run_tool(name, arguments)
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                        result = {"success": False, "error": str(error)}
                    await on_tool(name, result)
                    if name == "book_appointment" and result.get("success") is True:
                        confirmed_booking = result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result),
                        }
                    )
                if confirmed_booking is not None:
                    order_summary: dict[str, object] = {
                        "order_id": confirmed_booking["order_id"],
                        "name": confirmed_booking["name"],
                        "time": f"{confirmed_booking['time']} to {confirmed_booking['end_time']} on {confirmed_booking['date']}",
                        "charges_inr": int(confirmed_booking["charges_inr"]),
                    }
                    heading = "Here is your order summary:"
                    warning = str(confirmed_booking["payment_instructions"])
                    await on_sentence(heading)
                    if on_summary is not None:
                        await on_summary(order_summary)
                    await on_sentence(warning)
                    visible_response = "\n".join(
                        [
                            heading,
                            f"Order ID: {order_summary['order_id']}",
                            f"Name: {order_summary['name']}",
                            f"Time: {order_summary['time']}",
                            f"Charges: ₹{order_summary['charges_inr']:,}",
                            warning,
                        ]
                    )
                    history[:] = messages[1:]
                    return visible_response
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
