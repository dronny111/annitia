from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["trustii_id", "risk_hepatic_event", "risk_death"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ANNITIA submission CSV.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-rows", type=int, default=423)
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    if list(df.columns) != REQUIRED_COLUMNS:
        raise SystemExit(f"Expected columns {REQUIRED_COLUMNS}, got {list(df.columns)}")
    if len(df) != args.expected_rows:
        raise SystemExit(f"Expected {args.expected_rows} rows, got {len(df)}")
    if df["trustii_id"].duplicated().any():
        raise SystemExit("Duplicate trustii_id values found")
    if not df["trustii_id"].is_monotonic_increasing:
        raise SystemExit("Rows must be sorted by trustii_id")
    if df[["risk_hepatic_event", "risk_death"]].isna().any().any():
        raise SystemExit("Risk columns contain missing values")
    print(f"OK: {args.submission}")


if __name__ == "__main__":
    main()
