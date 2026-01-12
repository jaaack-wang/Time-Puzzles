import calendar
import datetime
import math
import os
import random
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
import pandas as pd
from lunarcalendar import Converter, Solar

from .utils import ordinal

# Configuration Constants
YEAR_CONTEXT_RANGE = 100

# Data Constants
LIFESPANS = {
    "Kobe Bryant": (datetime.date(1978, 8, 23), datetime.date(2020, 1, 26)),
    "Steve Jobs": (datetime.date(1955, 2, 24), datetime.date(2011, 10, 5)),
    "Michael Jackson": (datetime.date(1958, 8, 29), datetime.date(2009, 6, 25)),
    "Albert Einstein": (datetime.date(1879, 3, 14), datetime.date(1955, 4, 18)),
    "Elvis Presley": (datetime.date(1935, 1, 8), datetime.date(1977, 8, 16)),
    "Queen Elizabeth II": (datetime.date(1926, 4, 21), datetime.date(2022, 9, 8)),
}

PRESIDENTS = [
    ("Harry S. Truman", datetime.date(1945, 4, 12), datetime.date(1953, 1, 20)),
    ("Dwight D. Eisenhower", datetime.date(1953, 1, 20), datetime.date(1961, 1, 20)),
    ("John F. Kennedy", datetime.date(1961, 1, 20), datetime.date(1963, 11, 22)),
    ("Lyndon B. Johnson", datetime.date(1963, 11, 22), datetime.date(1969, 1, 20)),
    ("Richard Nixon", datetime.date(1969, 1, 20), datetime.date(1974, 8, 9)),
    ("Gerald Ford", datetime.date(1974, 8, 9), datetime.date(1977, 1, 20)),
    ("Jimmy Carter", datetime.date(1977, 1, 20), datetime.date(1981, 1, 20)),
    ("Ronald Reagan", datetime.date(1981, 1, 20), datetime.date(1989, 1, 20)),
    ("George H.W. Bush", datetime.date(1989, 1, 20), datetime.date(1993, 1, 20)),
    ("Bill Clinton", datetime.date(1993, 1, 20), datetime.date(2001, 1, 20)),
    ("George W. Bush", datetime.date(2001, 1, 20), datetime.date(2009, 1, 20)),
    ("Barack Obama", datetime.date(2009, 1, 20), datetime.date(2017, 1, 20)),
    ("Donald Trump", datetime.date(2017, 1, 20), datetime.date(2021, 1, 20)),
    ("Joe Biden", datetime.date(2021, 1, 20), datetime.date(2025, 1, 20)),
]


class FactLevel(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"


class Fact(ABC):
    """Abstract base class for a fact about a date."""

    @abstractmethod
    def level(self, meta: dict) -> FactLevel:
        """The granularity level of this fact (YEAR, MONTH, or DAY)."""
        pass

    @abstractmethod
    def generate(self, date: datetime.date) -> list[dict]:
        """Generate possible facts (metadata) for a given date."""
        pass

    @abstractmethod
    def validate(self, date: datetime.date, meta: dict) -> bool:
        """Check if a single date satisfies the fact."""
        pass

    @abstractmethod
    def format(self, meta: dict, explicit: bool = False) -> str:
        """Return natural language description."""
        pass

    def get_information_gain(self, meta: dict) -> float:
        """Estimate information gain in bits: -log2(probability)."""
        return 0.0

    @property
    def is_weekday_related(self) -> bool:
        return False


class YearRangeFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        year = date.year
        start_offset = random.randint(0, 100)
        start_year = year - start_offset
        end_year = start_year + 100

        return [
            {
                "type": "year_range",
                "start": int(start_year),
                "end": int(end_year),
            }
        ]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return meta["start"] <= date.year <= meta["end"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"The year is between {meta['start']} and {meta['end']} (inclusive)."

    def get_information_gain(self, meta: dict) -> float:
        # Range is 10 years. Assuming 100 year context.
        p = 10 / YEAR_CONTEXT_RANGE
        return -math.log2(p)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class DayOfYearFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        return [{"type": "day_of_year", "value": date.timetuple().tm_yday}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return date.timetuple().tm_yday == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"It is the {ordinal(meta['value'])} day of the year."

    def get_information_gain(self, meta: dict) -> float:
        # 1 day in 365
        return -math.log2(1 / 365.25)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class WeekdayInYearFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        facts = []
        weekday = date.weekday()  # 0=Mon, 6=Sun
        doy = date.timetuple().tm_yday
        year = date.year
        is_leap = calendar.isleap(year)
        total_days = 366 if is_leap else 365

        # Forward count
        # How many of this weekday have passed including today?
        # First occurrence of this weekday in the year:
        # Jan 1 is jan1_weekday.
        # First occurrence of target weekday is at offset (target - jan1 + 7) % 7
        # doy of first occurrence = 1 + offset
        # occurrence index = (doy - first_occurrence_doy) // 7 + 1

        jan1_weekday = datetime.date(year, 1, 1).weekday()
        offset = (weekday - jan1_weekday + 7) % 7
        first_occurrence_doy = 1 + offset

        if doy >= first_occurrence_doy:
            nth = (doy - first_occurrence_doy) // 7 + 1
            facts.append(
                {
                    "type": "weekday_in_year",
                    "direction": "forward",
                    "weekday": date.strftime("%A"),
                    "n": nth,
                }
            )

        # Backward count
        # Last occurrence logic similar
        # days_left = total_days - doy
        # nth_from_end = days_left // 7 + 1
        days_left = total_days - doy
        nth_from_end = days_left // 7 + 1
        facts.append(
            {
                "type": "weekday_in_year",
                "direction": "backward",
                "weekday": date.strftime("%A"),
                "n": nth_from_end,
            }
        )

        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        if date.strftime("%A") != meta["weekday"]:
            return False

        doy = date.timetuple().tm_yday
        year = date.year

        if meta["direction"] == "forward":
            jan1_weekday = datetime.date(year, 1, 1).weekday()
            target_weekday = date.weekday()
            offset = (target_weekday - jan1_weekday + 7) % 7
            first_occurrence_doy = 1 + offset
            if doy < first_occurrence_doy:
                return False
            nth = (doy - first_occurrence_doy) // 7 + 1
            return nth == meta["n"]
        else:
            is_leap = calendar.isleap(year)
            total_days = 366 if is_leap else 365
            days_left = total_days - doy
            nth_from_end = days_left // 7 + 1
            return nth_from_end == meta["n"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        if meta["direction"] == "forward":
            return f"It is the {ordinal(meta['n'])} {meta['weekday']} of the year."
        else:
            if meta["n"] == 1:
                return f"It is the last {meta['weekday']} of the year."
            else:
                return f"It is the {ordinal(meta['n'])} to last {meta['weekday']} of the year."

    def get_information_gain(self, meta: dict) -> float:
        # 1 day in 365
        return -math.log2(1 / 365.25)

    @property
    def is_weekday_related(self) -> bool:
        return True

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class WeekdayInMonthFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        facts = []
        day = date.day

        # Forward
        nth = (day - 1) // 7 + 1
        facts.append(
            {
                "type": "weekday_in_month",
                "direction": "forward",
                "weekday": date.strftime("%A"),
                "n": nth,
            }
        )

        # Backward
        days_in_month = calendar.monthrange(date.year, date.month)[1]
        days_left = days_in_month - day
        nth_from_end = days_left // 7 + 1
        facts.append(
            {
                "type": "weekday_in_month",
                "direction": "backward",
                "weekday": date.strftime("%A"),
                "n": nth_from_end,
            }
        )

        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        if date.strftime("%A") != meta["weekday"]:
            return False

        if meta["direction"] == "forward":
            nth = (date.day - 1) // 7 + 1
            return nth == meta["n"]
        else:
            days_in_month = calendar.monthrange(date.year, date.month)[1]
            days_left = days_in_month - date.day
            nth_from_end = days_left // 7 + 1
            return nth_from_end == meta["n"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        if meta["direction"] == "forward":
            return f"It is the {ordinal(meta['n'])} {meta['weekday']} of the month."
        else:
            if meta["n"] == 1:
                return f"It is the last {meta['weekday']} of the month."
            else:
                return f"It is the {ordinal(meta['n'])} to last {meta['weekday']} of the month."

    def get_information_gain(self, meta: dict) -> float:
        # 1 day in 30
        return -math.log2(1 / 30.44)

    @property
    def is_weekday_related(self) -> bool:
        return True

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class DayOfMonthFact(Fact):
    def _is_last_day_of_month(self, d: datetime.date) -> bool:
        next_day = d + datetime.timedelta(days=1)
        return next_day.day == 1

    def generate(self, date: datetime.date) -> list[dict]:
        facts = []
        # Specific Date
        facts.append({"type": "day_of_month_specific", "value": date.day})

        # First Day
        if date.day == 1:
            facts.append({"type": "first_day_of_month", "value": True})

        # Last Day
        if self._is_last_day_of_month(date):
            facts.append({"type": "last_day_of_month", "value": True})

        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        if meta["type"] == "day_of_month_specific":
            return date.day == meta["value"]
        elif meta["type"] == "first_day_of_month":
            return date.day == 1
        elif meta["type"] == "last_day_of_month":
            return self._is_last_day_of_month(date)
        return False

    def format(self, meta: dict, explicit: bool = False) -> str:
        if meta["type"] == "day_of_month_specific":
            return f"It is the {ordinal(meta['value'])} of the month."
        elif meta["type"] == "first_day_of_month":
            return "It is the first day of the month."
        elif meta["type"] == "last_day_of_month":
            return "It is the last day of the month."
        return ""

    def get_information_gain(self, meta: dict) -> float:
        # 1 day in 30
        return -math.log2(1 / 30.44)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class ExplicitYearFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        return [{"type": "explicit_year", "value": date.year}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return date.year == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"The year is {meta['value']}."

    def get_information_gain(self, meta: dict) -> float:
        # 1 year in 100
        return -math.log2(1 / YEAR_CONTEXT_RANGE)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class DecadeFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        decade = (date.year // 10) * 10
        return [{"type": "decade", "value": decade}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return (date.year // 10) * 10 == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"The year is in the {meta['value']}s."

    def get_information_gain(self, meta: dict) -> float:
        # 10 years in 100
        return -math.log2(10 / YEAR_CONTEXT_RANGE)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class WeekdayFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        return [{"type": "weekday", "value": date.strftime("%A")}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return date.strftime("%A") == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"It is a {meta['value']}."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 7)

    @property
    def is_weekday_related(self) -> bool:
        return True

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class MultiWeekdayFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        current_weekday = date.strftime("%A")
        all_weekdays = list(calendar.day_name)
        all_weekdays.remove(current_weekday)
        # Pick 1 or 2 other weekdays
        others = random.sample(all_weekdays, k=random.randint(1, 2))
        options = [current_weekday] + others
        random.shuffle(options)
        return [{"type": "multi_weekday", "value": options}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return date.strftime("%A") in meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        options = meta["value"]
        if len(options) == 2:
            return f"It is a {options[0]} or {options[1]}."
        else:
            return f"It is a {', '.join(options[:-1])}, or {options[-1]}."

    def get_information_gain(self, meta: dict) -> float:
        num_options = len(meta["value"])
        return -math.log2(num_options / 7)

    @property
    def is_weekday_related(self) -> bool:
        return True

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class DayOfMonthRangeFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        day = date.day
        facts = []
        # Before X
        if day < 28:
            upper_bound = random.randint(day + 1, 28)
            facts.append(
                {"type": "day_of_month", "operator": "<", "value": int(upper_bound)}
            )
        # After X
        if day > 1:
            lower_bound = random.randint(1, day - 1)
            facts.append(
                {"type": "day_of_month", "operator": ">", "value": int(lower_bound)}
            )
        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        if meta["operator"] == "<":
            return date.day < meta["value"]
        else:
            return date.day > meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        val = meta["value"]
        ord_val = ordinal(val)
        if meta["operator"] == "<":
            return f"It is before the {ord_val} of the month."
        else:
            return f"It is after the {ord_val} of the month."

    def get_information_gain(self, meta: dict) -> float:
        val = meta["value"]
        if meta["operator"] == "<":
            # Days 1 to val-1. Count = val - 1
            count = max(1, val - 1)
        else:
            # Days val+1 to 30. Count = 30 - val
            count = max(1, 30 - val)
        return -math.log2(count / 30.0)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.DAY


class MonthFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        month_name = date.strftime("%B")
        return [{"type": "month", "value": int(date.month), "name": month_name}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return date.month == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"It is {meta['name']}."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 12)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.MONTH


class LeapYearFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        if calendar.isleap(date.year):
            return [{"type": "event", "name": "leap_year"}]
        return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return calendar.isleap(date.year)

    def format(self, meta: dict, explicit: bool = False) -> str:
        return "It is a leap year."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 4)  # Approx

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class SeasonFact(Fact):
    def _get_season(self, d: datetime.date) -> str:
        month = d.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    def generate(self, date: datetime.date) -> list[dict]:
        season = self._get_season(date)
        return [{"type": "season", "value": season}]

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return self._get_season(date) == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"It is {meta['value']}."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 4)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.MONTH


class ChineseZodiacFact(Fact):
    def _get_chinese_zodiac(self, lunar_year: int) -> str:
        zodiacs = [
            "Rat",
            "Ox",
            "Tiger",
            "Rabbit",
            "Dragon",
            "Snake",
            "Horse",
            "Goat",
            "Monkey",
            "Rooster",
            "Dog",
            "Pig",
        ]
        return zodiacs[(lunar_year - 4) % 12]

    def generate(self, date: datetime.date) -> list[dict]:
        try:
            solar = Solar(date.year, date.month, date.day)
            lunar = Converter.Solar2Lunar(solar)
            zodiac = self._get_chinese_zodiac(lunar.year)
            return [{"type": "chinese_zodiac", "value": zodiac}]
        except Exception:
            return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        try:
            solar = Solar(date.year, date.month, date.day)
            lunar = Converter.Solar2Lunar(solar)
            return self._get_chinese_zodiac(lunar.year) == meta["value"]
        except Exception:
            return False

    def format(self, meta: dict, explicit: bool = False) -> str:
        return f"It is the Year of the {meta['value']} according to Chinese lunar calendar."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 12)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class LunarMonthFact(Fact):
    def generate(self, date: datetime.date) -> list[dict]:
        try:
            solar = Solar(date.year, date.month, date.day)
            lunar = Converter.Solar2Lunar(solar)
            return [{"type": "lunar_month", "value": int(lunar.month)}]
        except Exception:
            return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        try:
            solar = Solar(date.year, date.month, date.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar.month == meta["value"]
        except Exception:
            return False

    def format(self, meta: dict, explicit: bool = False) -> str:
        return (
            f"It is the {ordinal(meta['value'])} month of the Chinese lunar calendar."
        )

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 12)  # Approx

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.MONTH


class WesternZodiacFact(Fact):
    def _get_western_zodiac(self, month: int, day: int) -> str:
        if (month == 3 and day >= 22) or (month == 4 and day <= 18):
            return "Aries"
        if (month == 4 and day >= 21) or (month == 5 and day <= 19):
            return "Taurus"
        if (month == 5 and day >= 22) or (month == 6 and day <= 19):
            return "Gemini"
        if (month == 6 and day >= 22) or (month == 7 and day <= 21):
            return "Cancer"
        if (month == 7 and day >= 24) or (month == 8 and day <= 21):
            return "Leo"
        if (month == 8 and day >= 24) or (month == 9 and day <= 21):
            return "Virgo"
        if (month == 9 and day >= 24) or (month == 10 and day <= 21):
            return "Libra"
        if (month == 10 and day >= 24) or (month == 11 and day <= 20):
            return "Scorpio"
        if (month == 11 and day >= 23) or (month == 12 and day <= 20):
            return "Sagittarius"
        if (month == 12 and day >= 23) or (month == 1 and day <= 18):
            return "Capricorn"
        if (month == 1 and day >= 21) or (month == 2 and day <= 17):
            return "Aquarius"
        if (month == 2 and day >= 20) or (month == 3 and day <= 19):
            return "Pisces"
        return "Unknown"

    def generate(self, date: datetime.date) -> list[dict]:
        zodiac = self._get_western_zodiac(date.month, date.day)
        if zodiac != "Unknown":
            return [{"type": "western_zodiac", "value": zodiac}]
        return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return self._get_western_zodiac(date.month, date.day) == meta["value"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        if explicit:
            ranges = {
                "Aries": "March 22 and April 18",
                "Taurus": "April 21 and May 19",
                "Gemini": "May 22 and June 19",
                "Cancer": "June 22 and July 21",
                "Leo": "July 24 and August 21",
                "Virgo": "August 24 and September 21",
                "Libra": "September 24 and October 21",
                "Scorpio": "October 24 and November 20",
                "Sagittarius": "November 23 and December 20",
                "Capricorn": "December 23 and January 18",
                "Aquarius": "January 21 and February 17",
                "Pisces": "February 20 and March 19",
            }
            if meta["value"] in ranges:
                return f"The date is between {ranges[meta['value']]}."

        return f"The sun (zodiac) sign is {meta['value']}."

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 12)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.MONTH


class PersonAliveFact(Fact):
    def __init__(self, lifespans=None):
        self.lifespans = lifespans if lifespans else LIFESPANS

    def _is_alive(self, name: str, d: datetime.date) -> bool:
        if name not in self.lifespans:
            return False
        start, end = self.lifespans[name]
        # Strict check: avoid birth/death dates
        return (
            start + datetime.timedelta(days=1) <= d <= end - datetime.timedelta(days=1)
        )

    def generate(self, date: datetime.date) -> list[dict]:
        facts = []
        for name in self.lifespans.keys():
            if self._is_alive(name, date):
                facts.append({"type": "person_alive", "person": name})
        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return self._is_alive(meta["person"], date)

    def format(self, meta: dict, explicit: bool = False) -> str:
        if explicit:
            name = meta["person"]
            if name in self.lifespans:
                start, end = self.lifespans[name]
                s = start.strftime("%B %d, %Y")
                e = end.strftime("%B %d, %Y")
                return f"The date is between {s} and {e}."
        return f"{meta['person']} is alive."

    def get_information_gain(self, meta: dict) -> float:
        name = meta["person"]
        if name in self.lifespans:
            start, end = self.lifespans[name]
            days_alive = (end - start).days
            # Assuming 100 years context
            total_days = YEAR_CONTEXT_RANGE * 365.25
            p = days_alive / total_days
            return -math.log2(p)
        return 0.0

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class USPresidentFact(Fact):
    def _get_us_president(self, d: datetime.date) -> str:
        for name, start, end in PRESIDENTS:
            if (
                start + datetime.timedelta(days=1)
                <= d
                <= end - datetime.timedelta(days=1)
            ):
                return name
        return "Unknown"

    def generate(self, date: datetime.date) -> list[dict]:
        president = self._get_us_president(date)
        if president != "Unknown":
            return [{"type": "us_president", "person": president}]
        return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        return self._get_us_president(date) == meta["person"]

    def format(self, meta: dict, explicit: bool = False) -> str:
        if explicit:
            name = meta["person"]
            for pname, start, end in PRESIDENTS:
                if pname == name:
                    s = start.strftime("%B %d, %Y")
                    e = end.strftime("%B %d, %Y")
                    return f"The date is between {s} and {e}."

        return f"{meta['person']} is the President of the United States."

    def get_information_gain(self, meta: dict) -> float:
        # Find president term length
        name = meta["person"]
        term_days = 4 * 365.25  # Default approx
        for pname, start, end in PRESIDENTS:
            if pname == name:
                term_days = (end - start).days
                break
        total_days = YEAR_CONTEXT_RANGE * 365.25
        p = term_days / total_days
        return -math.log2(p)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class EventFact(Fact):
    def _is_summer_olympic_year(self, year: int) -> bool:
        if year < 1896:
            return False
        if year in {1916, 1940, 1944}:
            return False
        return (year - 1896) % 4 == 0

    def _is_world_cup_year(self, year: int) -> bool:
        if year < 1930:
            return False
        if year in {1942, 1946}:
            return False
        return (year - 1930) % 4 == 0

    def generate(self, date: datetime.date) -> list[dict]:
        facts = []
        if self._is_summer_olympic_year(date.year):
            facts.append({"type": "event", "name": "summer_olympics"})
        if self._is_world_cup_year(date.year):
            facts.append({"type": "event", "name": "world_cup"})
        return facts

    def validate(self, date: datetime.date, meta: dict) -> bool:
        if meta["name"] == "summer_olympics":
            return self._is_summer_olympic_year(date.year)
        elif meta["name"] == "world_cup":
            return self._is_world_cup_year(date.year)
        return False

    def format(self, meta: dict, explicit: bool = False) -> str:
        if explicit:
            if meta["name"] == "summer_olympics":
                return "The year satisfies year % 4 == 0, year >= 1896, and is not 1916, 1940, or 1944."
            elif meta["name"] == "world_cup":
                return "The year satisfies year % 4 == 2, year >= 1930, and is not 1942 or 1946."

        if meta["name"] == "summer_olympics":
            return "It is a modern Summer Olympic year."
        elif meta["name"] == "world_cup":
            return "It is a FIFA World Cup year."
        return ""

    def get_information_gain(self, meta: dict) -> float:
        return -math.log2(1 / 4)

    def level(self, meta: dict) -> FactLevel:
        return FactLevel.YEAR


class KnowledgeBaseEventFact(Fact):
    def __init__(self, pkl_path=None, year_range=(1950, 2025)):
        if pkl_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pkl_path = os.path.join(base_dir, "../data/facts.pkl")

        if not os.path.exists(pkl_path):
            # Fallback or error if pkl doesn't exist?
            # For now, let's assume it exists or try to generate it?
            # Better to just raise error or assume user ran preprocessing
            raise FileNotFoundError(
                f"Pickle file not found at {pkl_path}. Please run preprocess_facts.py first."
            )

        self.df = pd.read_pickle(pkl_path)
        self.year_range = year_range

    def _random_date_in_year(self, year):
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        delta = end - start
        return start + datetime.timedelta(days=random.randint(0, delta.days))

    def _random_date_in_month(self, month):
        year = self._get_random_year()
        _, days_in_month = calendar.monthrange(year, month)
        day = random.randint(1, days_in_month)
        return datetime.date(year, month, day)

    def _get_random_year(self):
        return random.randint(self.year_range[0], self.year_range[1])

    def _get_date_with_fixed_day(self, day):
        # Find a random date where the day of month is `day`
        # Try up to 20 times to find a valid month/year combination
        for _ in range(20):
            year = self._get_random_year()
            month = random.randint(1, 12)
            try:
                return datetime.date(year, month, day)
            except ValueError:
                continue
        # Fallback: January always has 31 days
        return datetime.date(self._get_random_year(), 1, day)

    def get_random_fact_and_date(self):
        # Pick a random row that has at least some data
        while True:
            row = self.df.sample(1).iloc[0]
            options = []

            # Define potential facts: (aspect, level, column_name)
            candidates = [
                ("start", FactLevel.YEAR, "start_year"),
                ("start", FactLevel.MONTH, "start_month"),
                ("start", FactLevel.DAY, "start_date"),
                ("end", FactLevel.YEAR, "end_year"),
                ("end", FactLevel.MONTH, "end_month"),
                ("end", FactLevel.DAY, "end_date"),
            ]

            for aspect, level, col in candidates:
                if pd.notna(row[col]):
                    options.append((aspect, level, int(row[col])))

            # Multi
            if isinstance(row["multi_years"], list) and row["multi_years"]:
                options.append(
                    ("multi_year", FactLevel.YEAR, random.choice(row["multi_years"]))
                )

            if options:
                break

        aspect, level, value = random.choice(options)

        # Generate Date
        if level == FactLevel.YEAR:
            date = self._random_date_in_year(value)
        elif level == FactLevel.MONTH:
            date = self._random_date_in_month(value)
        elif level == FactLevel.DAY:
            date = self._get_date_with_fixed_day(value)

        meta = {
            "type": "kb_event",
            "event": row["event"],
            "aspect": aspect,
            "level": level,
            "value": value,
        }

        return date, meta

    def generate(self, date: datetime.date) -> list[dict]:
        return []

    def validate(self, date: datetime.date, meta: dict) -> bool:
        level = meta["level"]
        value = meta["value"]

        if level == FactLevel.YEAR:
            if isinstance(value, list):  # multi_year
                return date.year in value
            return date.year == value
        elif level == FactLevel.MONTH:
            return date.month == value
        elif level == FactLevel.DAY:
            return date.day == value
        return False

    def format(self, meta: dict, explicit: bool = False) -> str:
        event = meta["event"]
        aspect = meta["aspect"]
        level = meta["level"]
        value = meta["value"]

        if explicit:
            if level == FactLevel.YEAR:
                if isinstance(value, list):
                    return f"The year is one of {','.join(map(str, value))}."
                return f"The year is {value}."
            elif level == FactLevel.MONTH:
                return f"The month is {calendar.month_name[value]}."
            elif level == FactLevel.DAY:
                return f"The day of the month is {value}."

        if aspect == "start":
            return f"It is the same {level.name.lower()} as the first day that {event}."
        elif aspect == "end":
            return f"It is the same {level.name.lower()} as the last day that {event}."
        elif aspect == "multi_year":
            return f"It is a year in which {event}."
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

    def level(self, meta: dict) -> FactLevel:
        return meta["level"]
