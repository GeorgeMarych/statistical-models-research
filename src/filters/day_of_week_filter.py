"""Day-of-week entry filters."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class DayOfWeekFilter(BaseFilter):
    """Allow entries only on selected weekdays, Monday=0 and Friday=4."""

    allowed_weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    label: str = "day_of_week_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"allowed_weekdays": list(self.allowed_weekdays)}

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        weekdays = pd.Series(data.index.weekday, index=data.index)
        mask = weekdays.isin(self.allowed_weekdays)
        return make_filter_frame(data, mask, mask, {"weekday": weekdays})
