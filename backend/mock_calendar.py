"""Small SQLite calendar with atomic appointment reservations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, time
from pathlib import Path
from collections.abc import Iterator


class CalendarStore:
    """Store one-hour turf bookings in the morning and evening sessions."""

    SLOT_MINUTES = 60
    SESSION_WINDOWS = {
        "morning": ((6 * 60, 11 * 60),),
        "evening": ((19 * 60, 24 * 60),),
    }

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
            raise ValueError("booking date must use YYYY-MM-DD format") from error

    @classmethod
    def _parse_time(cls, value: str) -> time:
        try:
            parsed = time.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ValueError("time must use HH:MM format") from error
        if parsed.second or parsed.microsecond:
            raise ValueError("turf slots must start on the hour")
        minute_of_day = parsed.hour * 60 + parsed.minute
        if not any(
            start <= minute_of_day < end and (minute_of_day - start) % cls.SLOT_MINUTES == 0
            for windows in cls.SESSION_WINDOWS.values()
            for start, end in windows
        ):
            raise ValueError("turf slots start hourly from 06:00-10:00 or 19:00-23:00")
        return parsed

    @classmethod
    def _range_windows(cls, time_range: str) -> tuple[tuple[int, int], ...]:
        normalized = time_range.strip().lower()
        if normalized == "morning":
            return cls.SESSION_WINDOWS["morning"]
        if normalized == "evening":
            return cls.SESSION_WINDOWS["evening"]
        if normalized in {"any", "all day"}:
            return cls.SESSION_WINDOWS["morning"] + cls.SESSION_WINDOWS["evening"]
        if "-" in normalized:
            start_text, end_text = normalized.split("-", maxsplit=1)
            try:
                start = time.fromisoformat(start_text.strip())
                end = time.fromisoformat(end_text.strip())
            except ValueError as error:
                raise ValueError("time_range must be morning, evening, any, or HH:MM-HH:MM") from error
            start_minute = start.hour * 60 + start.minute
            end_minute = 24 * 60 if end_text.strip() == "00:00" and start_minute > 0 else end.hour * 60 + end.minute
            if start.second or start.microsecond or end.second or end.microsecond:
                raise ValueError("custom turf windows must use whole-hour boundaries")
            if start_minute >= end_minute:
                raise ValueError("time_range must have an end later than its start")
            windows = tuple(
                (max(start_minute, session_start), min(end_minute, session_end))
                for session_start, session_end in cls.SESSION_WINDOWS["morning"] + cls.SESSION_WINDOWS["evening"]
                if max(start_minute, session_start) < min(end_minute, session_end)
            )
            if windows:
                return windows
            raise ValueError("custom turf windows must overlap 06:00-11:00 or 19:00-00:00")
        raise ValueError("time_range must be morning, evening, any, or HH:MM-HH:MM")

    @staticmethod
    def _price_for_slot(booking_date: date, start_time: time) -> int:
        if booking_date.weekday() >= 5:
            return 3000
        return 1500 if start_time.hour < 11 else 2500

    def check_availability(self, appointment_date: str, time_range: str) -> dict[str, object]:
        parsed_date = self._parse_date(appointment_date)
        windows = self._range_windows(time_range)

        with self._connect() as connection:
            booked_rows = connection.execute(
                "SELECT appointment_time FROM appointments WHERE appointment_date = ?",
                (parsed_date.isoformat(),),
            ).fetchall()
        booked = {row["appointment_time"] for row in booked_rows}

        slots = []
        for start_minute, end_minute in windows:
            for minute in range(start_minute, end_minute, self.SLOT_MINUTES):
                start_time = f"{minute // 60:02d}:{minute % 60:02d}"
                end_clock = (minute + self.SLOT_MINUTES) % (24 * 60)
                end_time = f"{end_clock // 60:02d}:{end_clock % 60:02d}"
                if start_time not in booked:
                    session = "morning" if start_minute < 11 * 60 else "evening"
                    slots.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "session": session,
                        }
                    )
        return {"date": parsed_date.isoformat(), "time_range": time_range, "available_slots": slots}

    def book_appointment(
        self,
        name: str,
        appointment_date: str,
        appointment_time: str,
        confirm_price: bool = False,
    ) -> dict[str, object]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("player name cannot be empty")
        parsed_date = self._parse_date(appointment_date).isoformat()
        parsed_date_value = date.fromisoformat(parsed_date)
        parsed_time_value = self._parse_time(appointment_time)
        parsed_time = parsed_time_value.strftime("%H:%M")
        end_minute = (parsed_time_value.hour * 60 + self.SLOT_MINUTES) % (24 * 60)
        end_time = f"{end_minute // 60:02d}:{end_minute % 60:02d}"
        charges_inr = self._price_for_slot(parsed_date_value, parsed_time_value)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT 1 FROM appointments WHERE appointment_date = ? AND appointment_time = ?",
                    (parsed_date, parsed_time),
                ).fetchone()
                if existing:
                    return {
                        "success": False,
                        "error": "That turf slot has just been booked. Please choose another available time.",
                        "date": parsed_date,
                        "time": parsed_time,
                    }
                if not confirm_price:
                    charge_text = f"₹{charges_inr:,}"
                    return {
                        "success": False,
                        "requires_price_confirmation": True,
                        "error": f"This one-hour turf slot costs {charge_text}. Confirm before I reserve it.",
                        "name": clean_name,
                        "date": parsed_date,
                        "time": parsed_time,
                        "end_time": end_time,
                        "session": "morning" if parsed_time_value.hour < 11 else "evening",
                        "charges_inr": charges_inr,
                    }
                cursor = connection.execute(
                    "INSERT INTO appointments (name, appointment_date, appointment_time) VALUES (?, ?, ?)",
                    (clean_name, parsed_date, parsed_time),
                )
        except sqlite3.IntegrityError:
            return {
                "success": False,
                "error": "That turf slot has just been booked. Please choose another available time.",
                "date": parsed_date,
                "time": parsed_time,
            }
        return {
            "success": True,
            "order_id": f"NT-{parsed_date.replace('-', '')}-{cursor.lastrowid:06d}",
            "confirmation_id": cursor.lastrowid,
            "name": clean_name,
            "date": parsed_date,
            "time": parsed_time,
            "end_time": end_time,
            "duration_minutes": self.SLOT_MINUTES,
            "charges_inr": charges_inr,
            "payment_instructions": (
                "Please arrive at the physical counter 10 minutes before your booking time and pay there. "
                "If payment is not completed, this turf slot may be assigned to someone else."
            ),
        }

    def list_appointments(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, appointment_date, appointment_time FROM appointments ORDER BY appointment_date, appointment_time"
            ).fetchall()
        return [dict(row) for row in rows]
