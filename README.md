# Platform Food Delivery and Urban Heat

This repository contains the analysis code for the study *Platform food delivery
redistributes urban heat exposure under extreme heat*. The study examines how food
delivery functions as an urban heat-adaptation service while potentially shifting
outdoor heat exposure from consumers to delivery riders. The analysis covers London,
Birmingham, Manchester, Leeds, and Glasgow during 2024. It combines platform order,
payment, and rider-operation records with gridded temperature, neighbourhood
population, demographic, and food-point-of-interest information.

The study evaluates several connected processes: nonlinear changes in delivery demand
as daily maximum temperature rises; changes in rider workload through order-taking
frequency and delivery distance; the relationship between workload and dynamic
delivery premiums; and differences in heat-exposure transfer across neighbourhood
social groups. Demand responses are estimated with natural cubic splines,
neighbourhood and date fixed effects, and moving-block bootstrap intervals.

## Code

- `01.py` prepares neighbourhood spatial, population, and food-POI context.
- `02.py` prepares daily temperature exposure and city heat thresholds.
- `03.py` calculates and validates the Delivery Performance Index and Heat Exposure
  Transfer Index.
- `04.py` estimates temperature-demand response curves and optional STL diagnostics.
- `05.py` downloads hourly ERA5 precipitation for selected dates.
- `06.py` produces the hourly precipitation diagnostic figure.

Together, these scripts support data preparation, variable construction, model
estimation, validation, and figure reproduction. Run them in numerical order where
applicable. Input paths and options are available through each script's `--help`
command.

## Installation

```bash
python -m pip install -r requirements.txt
```

Data are not included in this repository. The scripts use the platform and public
data sources described in the article.
