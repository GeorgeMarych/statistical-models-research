"""Reusable weekday entry filters."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.filters.base import BaseFilter, make_filter_frame

WEEKDAY_TO_INT = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass
class WeekdayFilter(BaseFilter):
    """Allow entries by weekday using readable day names."""

    skip_days: list[str] = field(default_factory=list)
    allowed_days: list[str] | None = None
    enabled: bool = True
    apply_to_long: bool = True
    apply_to_short: bool = True
    label: str = "weekday_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "skip_days": list(self.skip_days),
            "allowed_days": list(self.allowed_days) if self.allowed_days is not None else None,
            "enabled": self.enabled,
            "apply_to_long": self.apply_to_long,
            "apply_to_short": self.apply_to_short,
        }

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        weekdays = pd.Series(data.index.weekday, index=data.index)
        if not self.enabled:
            return make_filter_frame(data, True, True, {"weekday": weekdays})

        if self.allowed_days is not None:
            allowed = {_day_to_int(day) for day in self.allowed_days}
            mask = weekdays.isin(allowed)
        else:
            mask = pd.Series(True, index=data.index)

        if self.skip_days:
            skipped = {_day_to_int(day) for day in self.skip_days}
            mask &= ~weekdays.isin(skipped)

        allow_long = mask if self.apply_to_long else pd.Series(True, index=data.index)
        allow_short = mask if self.apply_to_short else pd.Series(True, index=data.index)
        return make_filter_frame(data, allow_long, allow_short, {"weekday": weekdays})


def _day_to_int(day: str | int) -> int:
    if isinstance(day, int):
        if day < 0 or day > 6:
            raise ValueError("weekday integers must be in 0..6")
        return day
    key = str(day).strip().lower()
    if key not in WEEKDAY_TO_INT:
        known = ", ".join(day.title() for day in WEEKDAY_TO_INT)
        raise ValueError(f"Unknown weekday {day!r}. Expected one of: {known}")
    return WEEKDAY_TO_INT[key]
