import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_date_str(date_str):
    if pd.isna(date_str):
        return np.nan, np.nan, np.nan
    parts = str(date_str).split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else np.nan
        day = int(parts[2]) if len(parts) > 2 else np.nan
        return year, month, day
    except ValueError:
        return np.nan, np.nan, np.nan


def parse_multi_year_spans(span_str):
    if pd.isna(span_str):
        return []
    years = set()
    spans = str(span_str).split(";")
    for span in spans:
        span = span.strip()
        if "~" in span:
            try:
                start, end = span.split("~")
                years.update(range(int(start), int(end) + 1))
            except ValueError:
                pass
        else:
            try:
                years.add(int(span))
            except ValueError:
                pass
    return sorted(list(years))


def preprocess_facts(input_csv, output_pkl):
    df = pd.read_csv(input_csv)

    # Apply parsing
    parsed_start = df["start"].apply(parse_date_str)
    df[["start_year", "start_month", "start_date"]] = pd.DataFrame(
        parsed_start.tolist(), index=df.index
    ).astype("Int64")

    parsed_end = df["end"].apply(parse_date_str)
    df[["end_year", "end_month", "end_date"]] = pd.DataFrame(
        parsed_end.tolist(), index=df.index
    ).astype("Int64")

    df["multi_years"] = df["multi_year_spans"].apply(parse_multi_year_spans)

    df.to_pickle(output_pkl)
    print(f"Processed data saved to {output_pkl}")

    all_years = pd.concat(
        [df["start_year"], df["end_year"], df["multi_years"].explode()]
    ).dropna()

    if not all_years.empty:
        print(f"Min year: {int(all_years.min())}")
        print(f"Max year: {int(all_years.max())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess facts from CSV to Pickle.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "fact_sheet.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "facts.pkl",
        help="Path to output Pickle file",
    )
    args = parser.parse_args()

    preprocess_facts(args.input, args.output)