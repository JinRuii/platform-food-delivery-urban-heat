#!/usr/bin/env python3
"""Aggregate neighbourhood HadUK-Grid tasmax and derive city heat thresholds.

English-city thresholds are the May-September 97.5th percentile of city-scale
daily maximum temperature, clipped to 26-28 C. Glasgow uses its unrounded annual
maximum, following the manuscript's separate maximum-temperature case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ENGLISH_CITIES = {"birmingham", "leeds", "london", "manchester"}
ALL_CITIES = ENGLISH_CITIES | {"glasgow"}
REPORTED_THRESHOLDS_C = {
    "london": 28.00,
    "birmingham": 27.62,
    "manchester": 26.46,
    "leeds": 26.26,
    "glasgow": 25.09,
}


def weighted_mean(group: pd.DataFrame) -> float:
    values = group["tasmax_mean_c"].to_numpy(float)
    weights = group["weight_area_m2"].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Temperature and area weights must be finite; weights must be positive")
    return float(np.average(values, weights=weights))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="five_cities_area_daily_tasmax_2024.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("derived/temperature"))
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(source, dtype={"area_code": str})
    required = {"city", "area_code", "date", "tasmax_mean_c", "weight_area_m2"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["city"] = data["city"].astype(str).str.strip().str.lower()
    unexpected = set(data["city"].unique()) - ALL_CITIES
    if unexpected:
        raise ValueError(f"Unexpected cities: {sorted(unexpected)}")
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise")
    if data.duplicated(["city", "area_code", "date"]).any():
        raise ValueError("Duplicate city-area-date rows found")
    for column in ("tasmax_mean_c", "weight_area_m2"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[list(required)].isna().any().any():
        raise ValueError("Required fields contain missing values")

    area_counts = data.groupby(["city", "date"])["area_code"].nunique()
    if area_counts.groupby(level=0).nunique().max() != 1:
        raise ValueError("The number of neighbourhoods changes within at least one city")
    city_daily = (
        data.groupby(["city", "date"], sort=True, observed=True)
        .apply(weighted_mean, include_groups=False)
        .rename("temperature_C").reset_index()
    )
    if city_daily.groupby("city")["date"].nunique().min() != 366:
        raise ValueError("Each city must contain all 366 dates of leap year 2024")

    threshold_rows = []
    for city, group in city_daily.groupby("city", sort=True):
        if city == "glasgow":
            raw = float(group["temperature_C"].max())
            threshold = raw
            rule = "annual maximum (unrounded selection value)"
        else:
            warm = group.loc[group["date"].dt.month.between(5, 9), "temperature_C"]
            raw = float(warm.quantile(0.975, interpolation="linear"))
            threshold = float(np.clip(raw, 26.0, 28.0))
            rule = "May-September 97.5th percentile, clipped to 26-28 C"
        threshold_rows.append({
            "city": city,
            "raw_threshold_C": raw,
            "analysis_threshold_C": threshold,
            "reported_manuscript_threshold_C": REPORTED_THRESHOLDS_C[city],
            "difference_from_reported_C": threshold - REPORTED_THRESHOLDS_C[city],
            "rule": rule,
            "city_days_at_or_above_threshold": int((group["temperature_C"] >= threshold).sum()),
            "annual_max_C": float(group["temperature_C"].max()),
        })

    thresholds = pd.DataFrame(threshold_rows)
    city_daily = city_daily.merge(
        thresholds[["city", "analysis_threshold_C"]], on="city", validate="many_to_one"
    )
    city_daily["is_high_temperature"] = (
        city_daily["temperature_C"] >= city_daily["analysis_threshold_C"]
    ).astype(int)
    city_daily["date"] = city_daily["date"].dt.strftime("%Y-%m-%d")
    city_daily.to_csv(output_dir / "city_daily_temperature.csv", index=False)
    thresholds.to_csv(output_dir / "city_heat_thresholds.csv", index=False)
    (output_dir / "temperature_processing_metadata.json").write_text(
        json.dumps({
            "source_file": source.name,
            "city_aggregation": "area-weighted mean of neighbourhood tasmax_mean_c",
            "area_weight": "weight_area_m2",
            "thresholds": threshold_rows,
            "note": (
                "Reported manuscript thresholds are included for audit only and are not "
                "substituted for thresholds calculated from the supplied area-temperature file."
            ),
        }, indent=2), encoding="utf-8"
    )
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
