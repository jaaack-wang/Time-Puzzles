DAY_CONSTRAINTS = {"day_of_month_specific", "weekday_in_month", 
                   "weekday", "day_of_month", "multi_weekday", 
                   "last_day_of_month", "first_day_of_month"}

MONTH_CONSTRAINTS = {"western_zodiac", "month", "lunar_month"}

YEAR_CONSTRAINTS = {"year_range", "explicit_year", "chinese_zodiac", 
                    "decade", "us_president", "event", "person_alive"}


def _count_time_constraints(constraints_meta: dict, 
                            constraint_types: set, 
                            constraint_type_str: str = None) -> int:
    count = 0
    for c in constraints_meta:
        t = c.get("type")
        if t in constraint_types:
            count += 1
        elif t == "kb_event" and constraint_type_str is not None:
            level = c.get("level")
            if level == constraint_type_str:
                count += 1
    return count


def count_day_constraints(constraints_meta: dict) -> int:
    return _count_time_constraints(constraints_meta, DAY_CONSTRAINTS, "day")


def count_month_constraints(constraints_meta: dict) -> int:
    return _count_time_constraints(constraints_meta, MONTH_CONSTRAINTS, "month")

def count_year_constraints(constraints_meta) -> int:
    return _count_time_constraints(constraints_meta, YEAR_CONSTRAINTS, "year")

