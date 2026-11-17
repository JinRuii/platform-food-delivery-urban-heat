#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate natural-cubic-spline demand responses with fixed effects.

Dependencies: python -m pip install numpy pandas scipy
Optional STL diagnostics also require: python -m pip install statsmodels
Usage: python 04.py --input orders.csv --out model_results

Input is a UTF-8 CSV with one neighbourhood-date-period per row and columns:
    date              Date (YYYY-MM-DD)
    city              City name
    neighbourhood_id  Globally unique neighbourhood identifier
    period            Lunch / Dinner / Other hours
    hours             Period duration: 2 / 2 / 4 hours
    temperature_C     City-day maximum temperature in degrees C
    orders            Non-negative integer order count for the row

The three mutually exclusive periods must cover the 11:30-19:30 operating
window. Lunch is 11:30-13:30, Other hours is 13:30-17:30 and Dinner is
17:30-19:30. The script does not convert 24-hour totals into this window and
does not turn missing observations into zeros. "All day" denotes this complete
eight-hour window, not a 24-hour day. Unbalanced neighbourhood-date panels are
allowed.

Model specification:
1. Sum the three periods for the eight-hour outcome and model log(orders/hours).
2. Analyse neighbourhood-days at T >= 15 C, using 15 C as the reference.
3. Use a natural cubic spline with two interior knots at temperature tertiles
   in the original warm sample. Boundaries are 15 C and the observed maximum.
4. Absorb neighbourhood and date fixed effects. City responses use city-by-spline
   interactions in the multi-city panel; period models use pooled spline knots.
5. By default, jointly resample dates with 500 seven-day moving-block bootstrap
   refits while keeping knots fixed.
6. Report percentage changes in conditional geometric mean hourly demand relative
   to 15 C, with pointwise 95% intervals on a 0.1 C grid and no extrapolation.

Outputs are estimated_spline_responses.csv and model_fit.json. Optional STL
diagnostics are summaries only: they are not passed to the main regression and
do not imply that the reported empirical curves were STL-detrended.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline

REFERENCE_C = 15.0
PERIODS = ["Lunch", "Dinner", "Other hours"]
HOURS = {"Lunch": 2, "Dinner": 2, "Other hours": 4, "All day": 8}
OPERATING_WINDOWS = {"Lunch": "11:30-13:30", "Dinner": "17:30-19:30",
                     "Other hours": "13:30-17:30", "All day": "11:30-19:30"}


def prepare_panel(raw):
    """Validate input and construct the neighbourhood-day analysis panel."""
    required = {"date", "city", "neighbourhood_id", "period", "hours", "temperature_C", "orders"}
    if not required <= set(raw):
        raise ValueError(f"Missing required columns: {sorted(required - set(raw))}")
    raw = raw[list(sorted(required))].copy()
    if raw.isna().any().any():
        raise ValueError("Required fields contain missing values; no automatic filling is applied.")
    for column in ["city", "neighbourhood_id", "period"]:
        raw[column] = raw[column].astype(str).str.strip()
        if raw[column].eq("").any():
            raise ValueError(f"{column} must not be blank.")
    parsed = pd.to_datetime(raw.date, format="%Y-%m-%d", errors="raise")
    if not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError("date must contain day-level dates.")
    raw["date"] = parsed.dt.strftime("%Y-%m-%d")
    for column in ["orders", "hours", "temperature_C"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if not np.isfinite(raw[["orders", "hours", "temperature_C"]].to_numpy(float)).all():
        raise ValueError("Orders, hours and temperature must be finite numeric values.")
    if (raw.orders < 0).any() or not np.equal(raw.orders, np.floor(raw.orders)).all():
        raise ValueError("orders must contain non-negative integer counts.")
    if set(raw.period) != set(PERIODS):
        raise ValueError("period must contain exactly Lunch, Dinner and Other hours.")
    if sum(HOURS[p] for p in PERIODS) != HOURS["All day"] or any(HOURS[p] <= 0 for p in PERIODS):
        raise ValueError("Period durations must be positive and sum to the eight-hour window.")
    if not np.isclose(raw.hours, raw.period.map(HOURS), rtol=0, atol=1e-9).all():
        raise ValueError("hours does not match the period definitions in HOURS.")
    if raw.groupby("neighbourhood_id").city.nunique().max() != 1:
        raise ValueError("neighbourhood_id must be globally unique across cities.")
    if raw.duplicated(["date", "neighbourhood_id", "period"]).any():
        raise ValueError("Duplicate neighbourhood-date-period records found.")
    if raw.groupby(["city", "date"]).temperature_C.nunique().max() != 1:
        raise ValueError("Temperature must be identical within each city-date.")
    keys = ["date", "city", "neighbourhood_id", "temperature_C"]
    if not raw.groupby(keys).period.nunique().eq(3).all():
        raise ValueError("Every neighbourhood-day must contain all three periods.")
    panel = raw.pivot(index=keys, columns="period", values="orders").reset_index()
    panel.columns.name = None
    panel["All day"] = panel[PERIODS].sum(axis=1)
    panel = panel.loc[panel.temperature_C >= REFERENCE_C].copy()
    if panel.empty or panel.city.nunique() < 2:
        raise ValueError("The warm sample must contain at least two cities.")
    outcomes = ["All day", "Lunch", "Dinner"]
    if (panel[outcomes] <= 0).any().any():
        raise ValueError("A model outcome contains zero orders; the script does not add one or drop rows.")
    for period in outcomes:
        panel[period] = np.log(panel[period] / HOURS[period])
    panel = panel[keys + outcomes].sort_values(["date", "city", "neighbourhood_id"]).reset_index(drop=True)
    # Draw date blocks from the full input calendar, before the warm-sample filter.
    panel.attrs["calendar_start"] = raw.date.min()
    panel.attrs["calendar_end"] = raw.date.max()
    return panel


def residualise(matrix, weights, groups, tolerance=1e-10, max_iter=1000):
    """Weighted alternating demeaning for two-way fixed effects (FWL)."""
    r = matrix.copy()
    for iteration in range(max_iter):
        maximum_adjustment = 0.0
        for g in groups:
            n = int(g.max()) + 1
            denominator = np.bincount(g, weights=weights, minlength=n)
            numerator = np.zeros((n, r.shape[1]))
            np.add.at(numerator, g, weights[:, None] * r)
            means = np.divide(numerator, denominator[:, None],
                              out=np.zeros_like(numerator), where=denominator[:, None] > 0)
            maximum_adjustment = max(maximum_adjustment, float(np.max(np.abs(means))))
            r -= means[g]
        if maximum_adjustment < tolerance:
            return r
    raise RuntimeError("Fixed-effects absorption did not converge.")


def solve_slopes(x, y, w):
    """OLS on absorbed variables; mark non-estimable individual coefficients NaN."""
    gram = x.T @ (w[:, None] * x)
    eigenvalues, vectors = eigh(gram, check_finite=False)
    keep = eigenvalues > max(float(eigenvalues[-1]) * 1e-10, 1e-12)
    if not keep.any():
        return np.full((x.shape[1], y.shape[1]), np.nan)
    vk = vectors[:, keep]
    estimate = vk @ ((vk.T @ (x.T @ (w[:, None] * y))) / eigenvalues[keep, None])
    if not keep.all():
        null_loading = np.sum(vectors[:, ~keep] ** 2, axis=1)
        estimate[null_loading > 1e-7] = np.nan
    return estimate


def make_spline_spec(temperatures):
    """Freeze knot locations using the ORIGINAL warm-sample temperatures."""
    t = np.asarray(temperatures, dtype=float)
    knots = np.quantile(t, [1 / 3, 2 / 3])
    nodes = np.r_[REFERENCE_C, knots, t.max()]
    if len(t) < 8 or not np.isfinite(nodes).all() or np.any(np.diff(nodes) <= 1e-8):
        raise ValueError("Two distinct interior tertile knots require adequate temperature support.")
    return {"interior_knots_C": knots.tolist(),
            "boundary_knots_C": [REFERENCE_C, float(t.max())],
            "observed_min_C": float(t.min()), "n_city_dates": int(len(t))}


def spline_basis(temperatures, spec):
    """Three natural-cubic regression columns, each centred at 15 C.

    Cardinal interpolation at the four fixed nodes spans the natural cubic
    spline space (four dimensions including the constant). Removing the first
    cardinal column after centring leaves three independent regression columns.
    CubicSpline constructs BASIS FUNCTIONS, not an interpolator through fitted
    demand points; coefficients below are estimated by fixed-effects OLS.
    No post-fit smoother, monotonicity constraint or extrapolation is applied.
    """
    lower, upper = spec["boundary_knots_C"]
    nodes = np.r_[lower, spec["interior_knots_C"], upper]
    basis = CubicSpline(nodes, np.eye(4), axis=0, bc_type="natural", extrapolate=False)
    t = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if np.any(t < lower - 1e-9) or np.any(t > upper + 1e-9):
        raise ValueError("Prediction outside the specified boundary knots is disabled.")
    return (basis(np.clip(t, lower, upper)) - basis(REFERENCE_C))[:, 1:]


def build_design(panel, cities):
    """Pooled design plus separate city spline blocks in the SAME panel."""
    support = panel[["city", "date", "temperature_C"]].drop_duplicates()
    specs = {"pooled": make_spline_spec(support.temperature_C)}
    common = spline_basis(panel.temperature_C, specs["pooled"])
    city_design = np.zeros((len(panel), 3 * len(cities)))
    for ci, city in enumerate(cities):
        mask = panel.city.eq(city).to_numpy()
        specs[city] = make_spline_spec(support.loc[support.city.eq(city), "temperature_C"])
        city_design[mask, 3*ci:3*(ci+1)] = spline_basis(
            panel.loc[mask, "temperature_C"], specs[city])
    return common, city_design, specs


def estimate(panel, bootstrap=500, block_days=7, seed=20261108):
    """Estimate two-way FE models and jointly resample moving date blocks.

    Estimation uses neighbourhood-day records and permits an unbalanced panel.
    All-day, lunch and dinner outcomes share regressors but are fitted separately.
    City responses are city-by-spline interactions in one multi-city panel. Each
    bootstrap refit re-absorbs fixed effects while retaining original-sample knots.
    Intervals are pointwise percentile intervals, not simultaneous bands.
    """
    if bootstrap < 30 or not 1 <= block_days <= 60:
        raise ValueError("Use >=30 replicates and a block length between 1 and 60 days.")
    cities_list = sorted(panel.city.unique())
    dates = pd.date_range(panel.attrs.get("calendar_start", panel.date.min()),
                          panel.attrs.get("calendar_end", panel.date.max()), freq="D")
    if len(dates) < block_days:
        raise ValueError("The sample calendar must be at least as long as the block length.")
    date_code = pd.Categorical(panel.date, categories=dates.strftime("%Y-%m-%d")).codes
    neighbourhood_code = pd.Categorical(panel.neighbourhood_id).codes
    if (date_code < 0).any():
        raise ValueError("Date coding failed.")
    common, interactions, specs = build_design(panel, cities_list)
    y = panel[["All day", "Lunch", "Dinner"]].to_numpy()
    joint = np.column_stack([common, interactions, y])
    outcome_start = 3 + 3 * len(cities_list)

    def fit(w):
        # Rows with zero bootstrap multiplicity are absent in this draw.
        selected = w > 0
        if not selected.any():
            return (np.full((3, 3), np.nan),
                    np.full((3 * len(cities_list), 1), np.nan))
        r = residualise(joint[selected], w[selected],
                        [neighbourhood_code[selected], date_code[selected]])
        ry = r[:, outcome_start:]
        return (solve_slopes(r[:, :3], ry, w[selected]),
                solve_slopes(r[:, 3:outcome_start], ry[:, :1], w[selected]))

    pooled, cities = fit(np.ones(len(panel)))
    if not np.isfinite(pooled).all() or not np.isfinite(cities).all():
        raise RuntimeError("Spline design is not identified after fixed-effects absorption.")
    rng = np.random.default_rng(seed)
    pooled_boot, cities_boot = [], []
    for b in range(bootstrap):
        starts = rng.integers(0, len(dates) - block_days + 1,
                              int(np.ceil(len(dates) / block_days)))
        selected = np.concatenate([np.arange(s, s + block_days) for s in starts])[:len(dates)]
        frequencies = np.bincount(selected, minlength=len(dates))
        pb, cb = fit(frequencies[date_code].astype(float))
        pooled_boot.append(pb)
        cities_boot.append(cb)
        if (b + 1) % 50 == 0:
            print(f"Bootstrap refits: {b+1}/{bootstrap}", flush=True)

    pooled_boot = np.asarray(pooled_boot)
    cities_boot = np.asarray(cities_boot)
    frames = []
    def summarise(model, group, spec, beta, draws):
        valid = np.isfinite(draws).all(axis=1)
        n_valid = int(valid.sum())
        if n_valid < max(20, int(np.ceil(0.8 * bootstrap))):
            raise RuntimeError(f"Insufficient identified bootstrap fits for {model}: {group}.")
        # This grid stops at the last 0.1 C point inside original-sample support.
        maximum = spec["boundary_knots_C"][1]
        grid = np.arange(int(REFERENCE_C*10), int(np.floor(maximum*10+1e-8))+1) / 10
        contrast = spline_basis(grid, spec)  # already B(T) - B(15)
        point_log = contrast @ beta
        boot_log = contrast @ draws[valid].T
        point_pct = 100 * np.expm1(point_log)
        boot_pct = 100 * np.expm1(boot_log)
        if not np.isfinite(boot_pct).all():
            raise RuntimeError("Non-finite bootstrap response; inspect the design and inputs.")
        lo, hi = np.quantile(boot_pct, [0.025, 0.975], axis=1)
        frames.append(pd.DataFrame({"model": model,
            "group": group, "temperature_C": grid, "log_response": point_log,
            "change_pct": point_pct, "ci_low_pct": lo, "ci_high_pct": hi,
            "valid_bootstrap": n_valid, "bootstrap_total": bootstrap,
            "n_city_dates": spec["n_city_dates"],
            "n_neighbourhood_dates": len(panel)}))

    summarise("Pooled all day", "All cities", specs["pooled"],
              pooled[:, 0], pooled_boot[:, :, 0])
    for ci, city in enumerate(cities_list):
        sl = slice(3*ci, 3*(ci+1))
        summarise("City differences", city, specs[city], cities[sl, 0], cities_boot[:, sl, 0])
    for i, period in enumerate(["All day", "Lunch", "Dinner"]):
        summarise("Period differences", period, specs["pooled"], pooled[:, i], pooled_boot[:, :, i])
    metadata = {"basis": "natural cubic spline, 3 nonconstant columns",
        "knots_fixed_across_bootstrap": True, "specifications": specs,
        "pooled_coefficients": pooled.tolist(), "city_coefficients": cities[:, 0].tolist(),
        "pooled_outcome_order": ["All day", "Lunch", "Dinner"],
        "city_coefficient_order": cities_list, "columns_per_city": 3}
    return pd.concat(frames, ignore_index=True), metadata



def stl_diagnostics(raw, period_days=7):
    """Optional STL summaries; demand-model outcomes remain unmodified.

    Decompose daily orders per operating hour for each neighbourhood and
    outcome BEFORE the warm-sample filter. Series must have consecutive dates
    over their observed span and at least two cycles. No interpolation, daily
    values or neighbourhood identifiers are exported. Summaries are aggregated
    by city and delivery period. A diagnostic is not evidence that the main
    empirical estimates were based on STL-detrended outcomes.
    """
    if period_days < 2:
        raise ValueError("The STL period must be at least two days.")
    try:
        from statsmodels.tsa.seasonal import STL
        import statsmodels
    except ImportError as exc:
        raise RuntimeError("Optional STL diagnostics require statsmodels.") from exc
    x = raw.copy()
    for col in ["city", "neighbourhood_id", "period"]:
        x[col] = x[col].astype(str).str.strip()
    x["date"] = pd.to_datetime(x.date, format="%Y-%m-%d")
    x["orders"] = pd.to_numeric(x.orders, errors="raise")
    wide = x.pivot(index=["city", "neighbourhood_id", "date"], columns="period", values="orders").reset_index()
    wide["All day"] = wide[PERIODS].sum(axis=1)
    rows = []
    for (city, neighbourhood), group in wide.groupby(["city", "neighbourhood_id"], sort=True):
        group = group.sort_values("date")
        dates = pd.DatetimeIndex(group.date)
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        if not dates.equals(expected) or len(group) < 2 * period_days:
            raise ValueError("STL diagnostics require consecutive daily series covering at least two cycles.")
        for outcome in ["All day", "Lunch", "Dinner"]:
            rate = group[outcome].to_numpy(float) / HOURS[outcome]
            decomposition = STL(rate, period=period_days, seasonal=7, robust=True).fit()
            detrended = rate - decomposition.trend
            rows.append({"city": city, "period": outcome, "n_dates": len(rate),
                         "trend_sd": float(np.std(decomposition.trend, ddof=1)),
                         "seasonal_sd": float(np.std(decomposition.seasonal, ddof=1)),
                         "residual_sd": float(np.std(decomposition.resid, ddof=1)),
                         "detrended_sd": float(np.std(detrended, ddof=1))})
    summaries = (pd.DataFrame(rows).groupby(["city", "period"], as_index=False)
        .agg(n_series=("n_dates", "size"), min_series_days=("n_dates", "min"),
             max_series_days=("n_dates", "max"), mean_trend_sd=("trend_sd", "mean"),
             mean_seasonal_sd=("seasonal_sd", "mean"), mean_residual_sd=("residual_sd", "mean"),
             mean_detrended_sd=("detrended_sd", "mean")))
    return {"enabled": True, "used_in_main_regression": False,
            "statsmodels_version": statsmodels.__version__, "period_days": period_days,
            "seasonal_smoother": 7, "robust": True, "trend_smoother": "statsmodels default",
            "scale": "orders per operating hour", "detrended_definition": "hourly_rate - STL_trend",
            "series_scope": "full input dates before temperature filtering",
            "summaries": summaries.to_dict(orient="records")}


def run(args):
    """Read aggregate platform orders, estimate models and save results."""
    if args.bootstrap < 30 or not 1 <= args.block_days <= 60:
        raise ValueError("Use at least 30 bootstrap refits and a block length from 1 to 60 days.")
    source = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    targets = [out / "estimated_spline_responses.csv", out / "model_fit.json"]
    if source in targets:
        raise ValueError("An output target equals the input path; choose another output directory.")
    raw = pd.read_csv(source, encoding="utf-8-sig", dtype={
        "date": str, "city": str, "neighbourhood_id": str, "period": str})
    panel = prepare_panel(raw)
    diagnostics = (stl_diagnostics(raw, args.stl_period) if args.stl_diagnostics
                   else {"enabled": False, "used_in_main_regression": False})
    results, metadata = estimate(panel, args.bootstrap, args.block_days, args.seed)
    metadata.update({
        "fixed_effects": ["neighbourhood_id", "date"],
        "estimation_unit": "neighbourhood-date; no city-date outcome collapse",
        "outcomes": "log(orders / period hours)",
        "sample_temperature_min_C": REFERENCE_C,
        "reference_temperature_C": REFERENCE_C,
        "response_formula": "100 * expm1(f_hat(T) - f_hat(15))",
        "response_interpretation": "conditional geometric mean hourly demand contrast",
        "bootstrap_replicates": args.bootstrap,
        "block_days": args.block_days,
        "bootstrap_seed": args.seed,
        "bootstrap_method": "joint non-circular moving calendar-date blocks",
        "ci_type": "pointwise 95% percentile; not simultaneous",
        "valid_fit_requirement": "at least max(20, ceil(0.8 * bootstrap)) per response",
        "period_hours": HOURS,
        "operating_windows_local_time": OPERATING_WINDOWS,
        "all_day_definition": "11:30-19:30 integrated 8-hour operating window",
        "stl_diagnostics": diagnostics,
        "n_neighbourhood_dates": len(panel),
        "n_neighbourhoods": int(panel.neighbourhood_id.nunique()),
        "n_cities": int(panel.city.nunique()),
        "n_observed_dates": int(panel.date.nunique()),
        "analysis_date_range": [panel.date.min(), panel.date.max()],
        "bootstrap_calendar_range": [panel.attrs["calendar_start"], panel.attrs["calendar_end"]],
        "prediction_step_C": 0.1,
        "zero_handling": "stop if a model outcome has zero orders in the warm sample",
    })
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(targets[0], index=False, encoding="utf-8-sig")
    targets[1].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Model estimation completed; response estimates and model metadata were saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the aggregate-order CSV")
    parser.add_argument("--out", default="model_results", help="Output directory")
    parser.add_argument("--bootstrap", type=int, default=500, help="Number of bootstrap refits")
    parser.add_argument("--block-days", type=int, default=7, help="Moving-block length in days")
    parser.add_argument("--seed", type=int, default=20261108, help="Bootstrap random seed")
    parser.add_argument("--stl-diagnostics", action="store_true",
                        help="Add optional STL summary diagnostics without changing the regression")
    parser.add_argument("--stl-period", type=int, default=7, help="STL period in days")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Model did not complete: {exc}\n")
