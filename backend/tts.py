"""Microsoft Edge neural TTS adapter and spoken-text cleanup."""

from __future__ import annotations

import asyncio
import re
from datetime import date


_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CLOCK_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def prepare_speech_text(text: str) -> str:
    """Remove markup and make dates, times, and symbols easier to speak."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    def format_date(match: re.Match[str]) -> str:
        try:
            parsed = date.fromisoformat(match.group(0))
        except ValueError:
            return match.group(0)
        return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"

    def format_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        suffix = "AM" if hour < 12 else "PM"
        spoken_hour = hour % 12 or 12
        return f"{spoken_hour}:{minute:02d} {suffix}" if minute else f"{spoken_hour} {suffix}"

    text = _ISO_DATE.sub(format_date, text)
    text = _CLOCK_TIME.sub(format_time, text)
    text = re.sub(r"₹\s*([\d,]+(?:\.\d{1,2})?)", r"\1 rupees", text)
    text = text.replace("&", " and ").replace("—", " to ").replace("–", " to ")
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", text)
    text = re.sub(r"[*_`#|{}<>\[\]]", " ", text)
    text = re.sub(r"[^\w\s.,?!:;’'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


class EdgeTtsSynthesizer:
    def __init__(
        self,
        voice_name: str = "en-US-AriaNeural",
        rate: str = "+8%",
    ) -> None:
        self.voice_name = voice_name
        self.rate = rate

    def synthesize(self, text: str) -> bytes:
        speech_text = prepare_speech_text(text)
        if not speech_text:
            return b""

        async def collect_audio() -> bytes:
            import edge_tts

            audio_chunks = []
            communicator = edge_tts.Communicate(speech_text, self.voice_name, rate=self.rate)
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            if not audio_chunks:
                raise RuntimeError("Edge TTS returned no audio")
            return b"".join(audio_chunks)

        return asyncio.run(collect_audio())
