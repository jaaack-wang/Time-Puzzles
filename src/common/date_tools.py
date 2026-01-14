import re

MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

ISO_PATTERN = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
US_NUMERIC_PATTERN = re.compile(r"\b(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{4})\b")
WORD_MONTH_PATTERN = re.compile(
    r"\b(?P<month_name>"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
    r")\s+(?P<day>\d{1,2})(?:,)?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

def is_leap_year(year: int) -> bool:
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)

def is_valid_date(year: int, month: int, day: int) -> bool:
    if not (1 <= month <= 12):
        return False

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month == 2 and is_leap_year(year):
        max_day = 29
    else:
        max_day = days_in_month[month - 1]

    return 1 <= day <= max_day

def extract_dates(text: str):
    results = []
    patterns = [ISO_PATTERN, US_NUMERIC_PATTERN, WORD_MONTH_PATTERN]

    for pattern in patterns:
        for m in pattern.finditer(text):
            if "month_name" in m.groupdict() and m.group("month_name") is not None:
                name = m.group("month_name").lower()
                month = MONTH_NAMES.get(name)
                if month is None:
                    continue
            else:
                month = int(m.group("month"))

            year = int(m.group("year"))
            day = int(m.group("day"))

            if is_valid_date(year, month, day):
                results.append({
                    "year": year,
                    "month": month,
                    "day": day,
                    "original": m.group(0)
                })

    return results
