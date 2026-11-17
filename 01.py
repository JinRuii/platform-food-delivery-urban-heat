#!/usr/bin/env python3
"""Build neighbourhood population and food-POI context for the five study cities.

The script reads the public boundary shapefiles, public population tables and
OpenStreetMap food-POI CSV files already downloaded by the research team. It
performs a point-in-polygon join and writes area-level counts and densities.
It does not read or export any platform order, payment or rider records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


CITIES = {
    "birmingham": "birmingham_msoa_2021.shp",
    "glasgow": "glasgow_intermediate_zones_2022.shp",
    "leeds": "leeds_msoa_2021.shp",
    "london": "london_msoa_2021.shp",
    "manchester": "manchester_msoa_2021.shp",
}
CODE_CANDIDATES = (
    "area_code", "MSOA21CD", "msoa21cd", "IZ_CODE", "IZCode", "InterZone", "interzone",
)
NAME_CANDIDATES = (
    "area_name", "MSOA21NM", "msoa21nm", "IZ_NAME", "IZName", "InterZoneN", "interzon2",
)


def select_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Return the first exact or case-insensitive candidate present in a table."""
    exact = set(frame.columns)
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not identify {label}; available columns: {list(frame.columns)}")


def process_city(data_dir: Path, city: str, boundary_name: str) -> tuple[gpd.GeoDataFrame, dict]:
    boundary_path = data_dir / boundary_name
    population_path = data_dir / f"{city}_population_by_area.csv"
    poi_path = data_dir / f"{city}_food_poi_osm.csv"
    for path in (boundary_path, population_path, poi_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    boundaries = gpd.read_file(boundary_path)
    if boundaries.empty or boundaries.crs is None:
        raise ValueError(f"Boundary file is empty or has no CRS: {boundary_path}")
    code_column = select_column(boundaries, CODE_CANDIDATES, "area-code column")
    name_column = select_column(boundaries, NAME_CANDIDATES, "area-name column")
    boundaries = boundaries[[code_column, name_column, "geometry"]].rename(
        columns={code_column: "area_code", name_column: "area_name"}
    )
    boundaries["area_code"] = boundaries["area_code"].astype(str)
    if boundaries["area_code"].duplicated().any():
        raise ValueError(f"Duplicate area codes in {boundary_path}")

    population = pd.read_csv(population_path, dtype={"area_code": str})
    required_population = {"area_code", "area_name"}
    missing = required_population - set(population.columns)
    if missing:
        raise ValueError(f"Missing population columns in {population_path}: {sorted(missing)}")
    population_column = select_column(population, ("pop_2021", "pop_2022"), "population column")
    population = population.rename(columns={population_column: "population"})
    if population["area_code"].duplicated().any() or population["population"].isna().any():
        raise ValueError(f"Population keys must be unique and complete in {population_path}")

    areas = boundaries.merge(
        population[["area_code", "population"]], on="area_code", how="left", validate="one_to_one"
    )
    if areas["population"].isna().any():
        missing_codes = areas.loc[areas["population"].isna(), "area_code"].tolist()
        raise ValueError(f"Population is missing for boundary codes: {missing_codes[:10]}")

    poi = pd.read_csv(poi_path, dtype={"osmid": str})
    required_poi = {"longitude", "latitude", "poi_category"}
    missing = required_poi - set(poi.columns)
    if missing:
        raise ValueError(f"Missing POI columns in {poi_path}: {sorted(missing)}")
    coordinates = poi[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce")
    valid = coordinates.notna().all(axis=1)
    if not valid.all():
        raise ValueError(f"Invalid POI coordinates in {poi_path}: {int((~valid).sum())} rows")
    poi_points = gpd.GeoDataFrame(
        poi.copy(),
        geometry=gpd.points_from_xy(coordinates["longitude"], coordinates["latitude"]),
        crs="EPSG:4326",
    ).to_crs(areas.crs)

    joined = gpd.sjoin(
        poi_points[["poi_category", "geometry"]],
        areas[["area_code", "geometry"]],
        how="left",
        predicate="within",
    )
    counts = joined.dropna(subset=["area_code"]).groupby("area_code").size().rename("food_poi_count")
    category_counts = (
        joined.dropna(subset=["area_code"])
        .groupby(["area_code", "poi_category"]).size().unstack(fill_value=0)
        .add_prefix("poi_").add_suffix("_count")
    )
    areas = areas.join(counts, on="area_code").join(category_counts, on="area_code")
    count_columns = [column for column in areas.columns if column.endswith("_count")]
    areas[count_columns] = areas[count_columns].fillna(0).astype(int)

    projected = areas.to_crs("EPSG:27700")
    areas["area_km2"] = projected.geometry.area.to_numpy() / 1_000_000
    if (areas["area_km2"] <= 0).any():
        raise ValueError(f"Non-positive polygon area found for {city}")
    areas["population_density_km2"] = areas["population"] / areas["area_km2"]
    areas["food_poi_density_km2"] = areas["food_poi_count"] / areas["area_km2"]
    areas.insert(0, "city", city)

    summary = {
        "city": city,
        "areas": int(len(areas)),
        "population": int(areas["population"].sum()),
        "food_pois_input": int(len(poi_points)),
        "food_pois_inside_boundaries": int(joined["area_code"].notna().sum()),
        "food_pois_outside_boundaries": int(joined["area_code"].isna().sum()),
    }
    return areas, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing public input files")
    parser.add_argument("--output-dir", type=Path, default=Path("derived/spatial_context"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, summaries = [], []
    for city, boundary_name in CITIES.items():
        frame, summary = process_city(data_dir, city, boundary_name)
        frames.append(pd.DataFrame(frame.drop(columns="geometry")))
        summaries.append(summary)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    numeric = combined.select_dtypes(include=[np.number]).columns
    if not np.isfinite(combined[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite numeric values found in the derived spatial context")
    combined.to_csv(output_dir / "neighbourhood_spatial_context.csv", index=False)
    (output_dir / "spatial_context_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
