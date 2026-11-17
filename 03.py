#!/usr/bin/env python3
"""Recalculate DPI and HETI from neighbourhood-day platform data.

The script verifies the published variable-construction equations against the
stored period-specific columns. Platform data and derived row-level outputs are
excluded by .gitignore; only code is intended for GitHub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PERIOD_PREFIXES = ("all_day", "lunch_peak", "dinner_peak", "daytime_window")
IDENTIFIERS = (
    "date", "city", "geo_id", "population", "income_group",
    "education_group", "dominant_age_group", "tasmax_local_c",
)


def required_columns() -> set[str]:
    columns = set(IDENTIFIERS)
    for prefix in PERIOD_PREFIXES:
        columns.update({
            f"{prefix}_delta_order_intensity_oph",
            f"{prefix}_orders_per_rider_hour",
            f"{prefix}_avg_delivery_distance_km",
            f"{prefix}_dpi",
            f"{prefix}_heti",
        })
    return columns


def process_chunk(chunk: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, list[dict]]:
    missing = required_columns() - set(chunk.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    output = chunk[list(IDENTIFIERS)].copy()
    if chunk[list(IDENTIFIERS)].isna().any().any():
        raise ValueError("Identifier or grouping columns contain missing values")
    population = pd.to_numeric(chunk["population"], errors="raise")
    if (population <= 0).any():
        raise ValueError("population must be positive")

    checks = []
    for prefix in PERIOD_PREFIXES:
        frequency = pd.to_numeric(chunk[f"{prefix}_orders_per_rider_hour"], errors="raise")
        distance = pd.to_numeric(chunk[f"{prefix}_avg_delivery_distance_km"], errors="raise")
        delta = pd.to_numeric(chunk[f"{prefix}_delta_order_intensity_oph"], errors="raise")
        stored_dpi = pd.to_numeric(chunk[f"{prefix}_dpi"], errors="raise")
        stored_heti = pd.to_numeric(chunk[f"{prefix}_heti"], errors="raise")
        arrays = np.column_stack([frequency, distance, delta, stored_dpi, stored_heti, population])
        if not np.isfinite(arrays).all():
            raise ValueError(f"Non-finite values found for period prefix {prefix}")

        calculated_dpi = frequency * distance
        # The stored files express HETI per 1,000 residents.
        calculated_heti = delta * calculated_dpi / population * 1_000
        dpi_error = (stored_dpi - calculated_dpi).abs()
        heti_error = (stored_heti - calculated_heti).abs()
        checks.append({
            "period_prefix": prefix,
            "rows": int(len(chunk)),
            "maximum_absolute_dpi_difference": float(dpi_error.max()),
            "maximum_absolute_heti_difference": float(heti_error.max()),
            "dpi_rows_outside_tolerance": int(dpi_error.gt(tolerance).sum()),
            "heti_rows_outside_tolerance": int(heti_error.gt(tolerance).sum()),
        })
        output[f"{prefix}_dpi_recalculated"] = calculated_dpi
        output[f"{prefix}_heti_per_1000_recalculated"] = calculated_heti
    return output, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing neighbourhood daily platform-data CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("private_derived/workload_heti"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--tolerance", type=float, default=6e-4,
                        help="Absolute tolerance for rounded stored columns")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_delivery_neighbourhood_daily_*.csv"))
    if not files:
        raise FileNotFoundError("No neighbourhood daily CSV files matched the expected pattern")
    summary = []
    output_path = output_dir / "recalculated_workload_heti.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to append to an existing output: {output_path}")
    wrote_header = False
    seen_keys: set[tuple[str, str]] = set()
    for source in files:
        file_rows = 0
        file_checks: list[dict] = []
        for chunk in pd.read_csv(source, chunksize=args.chunksize):
            derived, checks = process_chunk(chunk, args.tolerance)
            if derived.duplicated(["date", "geo_id"]).any():
                raise ValueError(f"Duplicate date-geo_id keys within {source.name}")
            keys = list(zip(derived["date"].astype(str), derived["geo_id"].astype(str)))
            duplicates = seen_keys.intersection(keys)
            if duplicates:
                raise ValueError(f"Duplicate date-geo_id keys across inputs: {list(duplicates)[:5]}")
            seen_keys.update(keys)
            derived.to_csv(output_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            file_rows += len(derived)
            file_checks.extend(checks)
        aggregated_checks = []
        for prefix in PERIOD_PREFIXES:
            period_checks = [item for item in file_checks if item["period_prefix"] == prefix]
            aggregated_checks.append({
                "period_prefix": prefix,
                "rows": sum(item["rows"] for item in period_checks),
                "maximum_absolute_dpi_difference": max(
                    item["maximum_absolute_dpi_difference"] for item in period_checks
                ),
                "maximum_absolute_heti_difference": max(
                    item["maximum_absolute_heti_difference"] for item in period_checks
                ),
                "dpi_rows_outside_tolerance": sum(
                    item["dpi_rows_outside_tolerance"] for item in period_checks
                ),
                "heti_rows_outside_tolerance": sum(
                    item["heti_rows_outside_tolerance"] for item in period_checks
                ),
            })
        summary.append({"input_file": source.name, "rows": file_rows, "checks": aggregated_checks})

    metadata = {
        "input_data_status": "platform data",
        "rows": sum(item["rows"] for item in summary),
        "dpi_equation": "orders_per_rider_hour * average_delivery_distance_km",
        "heti_equation": "delta_order_intensity_oph * DPI / population * 1000",
        "heti_scale": "per 1,000 residents in the stored analytical files",
        "tolerance": args.tolerance,
        "files": summary,
    }
    (output_dir / "workload_heti_validation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
