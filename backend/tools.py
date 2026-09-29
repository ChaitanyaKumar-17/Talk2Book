"""Function tools exposed to the voice agent."""

from __future__ import annotations

from backend.mock_calendar import CalendarStore


def check_availability(calendar: CalendarStore, date: str, time_range: str) -> dict[str, object]:
    return calendar.check_availability(date, time_range)


def book_appointment(
    calendar: CalendarStore,
    name: str,
    date: str,
    time: str,
) -> dict[str, object]:
    return calendar.book_appointment(name, date, time)
