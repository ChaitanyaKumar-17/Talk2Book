"""Sentence-at-a-time Kokoro synthesis."""

from __future__ import annotations

from io import BytesIO


class KokoroSynthesizer:
    def __init__(self, voice: str = "af_heart") -> None:
        self.voice = voice
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="a")
        return self._pipeline

    def synthesize(self, text: str) -> bytes:
        import soundfile as sf

        audio_parts = [audio for _, _, audio in self._get_pipeline()(text, voice=self.voice)]
        if not audio_parts:
            return b""
        import numpy as np

        audio = np.concatenate(audio_parts)
        output = BytesIO()
        sf.write(output, audio, 24000, format="WAV", subtype="PCM_16")
        return output.getvalue()
