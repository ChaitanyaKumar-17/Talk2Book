"""Small SQLite calendar with atomic appointment reservations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, time
from pathlib import Path
from collections.abc import Iterator


class CalendarStore:
    """Store booked half-hour appointments in a local SQLite database."""

    OPENING_MINUTE = 9 * 60
    CLOSING_MINUTE = 17 * 60
    SLOT_MINUTES = 30

    def __init__(self, database_path: str | Path = "talk2book.sqlite3") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    appointment_time TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (appointment_date, appointment_time)
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def _parse_date(cls, value: str) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ValueError("date must use YYYY-MM-DD format") from error

    @classmethod
    def _parse_time(cls, value: str) -> time:
        try:
            parsed = time.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ValueError("time must use HH:MM format") from error
        if parsed.second or parsed.microsecond:
            raise ValueError("appointments must start on a half-hour")
        minute_of_day = parsed.hour * 60 + parsed.minute
        if (
            minute_of_day < cls.OPENING_MINUTE
            or minute_of_day >= cls.CLOSING_MINUTE
            or minute_of_day % cls.SLOT_MINUTES
        ):
            raise ValueError("appointments are available every half-hour from 09:00 to 16:30")
        return parsed

    @classmethod
    def _range_bounds(cls, time_range: str) -> tuple[int, int]:
        normalized = time_range.strip().lower()
        named_ranges = {
            "morning": (9 * 60, 12 * 60),
            "afternoon": (12 * 60, 17 * 60),
            "evening": (15 * 60, 17 * 60),
            "any": (cls.OPENING_MINUTE, cls.CLOSING_MINUTE),
            "all day": (cls.OPENING_MINUTE, cls.CLOSING_MINUTE),
        }
        if normalized in named_ranges:
            return named_ranges[normalized]
        if "-" in normalized:
            start_text, end_text = normalized.split("-", maxsplit=1)
            try:
                start = time.fromisoformat(start_text.strip())
                end = time.fromisoformat(end_text.strip())
            except ValueError as error:
                raise ValueError("time_range must be morning, afternoon, evening, or HH:MM-HH:MM") from error
            start_minute = start.hour * 60 + start.minute
            end_minute = end.hour * 60 + end.minute
            return max(start_minute, cls.OPENING_MINUTE), min(end_minute, cls.CLOSING_MINUTE)
        raise ValueError("time_range must be morning, afternoon, evening, or HH:MM-HH:MM")

    def check_availability(self, appointment_date: str, time_range: str) -> dict[str, object]:
        parsed_date = self._parse_date(appointment_date)
        start_minute, end_minute = self._range_bounds(time_range)
        if start_minute >= end_minute:
            raise ValueError("time_range must have an end later than its start")

        with self._connect() as connection:
            booked_rows = connection.execute(
                "SELECT appointment_time FROM appointments WHERE appointment_date = ?",
                (parsed_date.isoformat(),),
            ).fetchall()
        booked = {row["appointment_time"] for row in booked_rows}

        slots = []
        for minute in range(self.OPENING_MINUTE, self.CLOSING_MINUTE, self.SLOT_MINUTES):
            if start_minute <= minute < end_minute:
                slot = f"{minute // 60:02d}:{minute % 60:02d}"
                if slot not in booked:
                    slots.append(slot)
        return {"date": parsed_date.isoformat(), "time_range": time_range, "available_slots": slots}

    def book_appointment(self, name: str, appointment_date: str, appointment_time: str) -> dict[str, object]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name cannot be empty")
        parsed_date = self._parse_date(appointment_date).isoformat()
        parsed_time = self._parse_time(appointment_time).strftime("%H:%M")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "INSERT INTO appointments (name, appointment_date, appointment_time) VALUES (?, ?, ?)",
                    (clean_name, parsed_date, parsed_time),
                )
        except sqlite3.IntegrityError:
            return {
                "success": False,
                "error": "That slot has just been booked. Please choose another available time.",
                "date": parsed_date,
                "time": parsed_time,
            }
        return {
            "success": True,
            "confirmation_id": cursor.lastrowid,
            "name": clean_name,
            "date": parsed_date,
            "time": parsed_time,
        }

    def list_appointments(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, appointment_date, appointment_time FROM appointments ORDER BY appointment_date, appointment_time"
            ).fetchall()
        return [dict(row) for row in rows]
