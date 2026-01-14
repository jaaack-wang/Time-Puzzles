from time import time
from typing import Callable


def seconds_to_human_readable(total_seconds: int) -> str:
    """
    Convert seconds into a human-readable format: D-H-M-S.
    """
    days = int(total_seconds // 86400)
    total_seconds %= 86400
    
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    
    minutes = int(total_seconds // 60)
    seconds = round(total_seconds % 60, 2)

    return f"{days}d-{hours}h-{minutes}m-{seconds}s"


def time_experiment_in_human_readable(experiment: Callable, **kwargs) -> None:
    """
    Time an experiment in a human readable time format.
    """
    start = time(); experiment(**kwargs); end = time()
    duration = seconds_to_human_readable(end-start)
    print(f"Execution finished. Total duration: {duration}")