"""Persist per-turn time-to-first-audio measurements as JSON."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4


class LatencyRecorder:
    def __init__(self, report_path: str | Path) -> None:
        self.report_path = Path(report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def start_turn(self) -> dict[str, float | str]:
        return {"turn_id": str(uuid4()), "speech_ended_at": perf_counter()}

    def finish_turn(
        self,
        timing: dict[str, float | str],
        *,
        llm_started_at: float | None,
        first_token_at: float | None,
        first_audio_at: float | None,
        tts_seconds: float,
    ) -> dict[str, object]:
        speech_ended_at = float(timing["speech_ended_at"])
        result: dict[str, object] = {
            "turn_id": timing["turn_id"],
            "time_to_first_audio_ms": round((first_audio_at - speech_ended_at) * 1000, 1)
            if first_audio_at is not None
            else None,
            "stt_to_llm_ms": round((llm_started_at - speech_ended_at) * 1000, 1)
            if llm_started_at is not None
            else None,
            "llm_time_to_first_token_ms": round((first_token_at - llm_started_at) * 1000, 1)
            if first_token_at is not None and llm_started_at is not None
            else None,
            "tts_synthesis_seconds": round(tts_seconds, 3),
        }
        with self._lock:
            try:
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {"turns": []}
            turns = report.setdefault("turns", [])
            turns.append(result)
            latencies = sorted(
                turn["time_to_first_audio_ms"]
                for turn in turns
                if turn.get("time_to_first_audio_ms") is not None
            )
            report["summary"] = {
                "completed_turns": len(latencies),
                "median_time_to_first_audio_ms": round(median(latencies), 1) if latencies else None,
                "p95_time_to_first_audio_ms": round(latencies[math.ceil(len(latencies) * 0.95) - 1], 1)
                if latencies
                else None,
            }
            self.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return result
