"""Download hourly ERA5 precipitation for the five peak-date entries in Supplementary Table 2.

Uses only Python's standard library. The data are reanalysis grid estimates, not station observations.
"""
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from decimal import Decimal
import csv
import json
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'raw_api').mkdir(exist_ok=True)
UK = ZoneInfo('Europe/London')
BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'
CASES = [
    ('London', 51.5074, -0.1278, '2024-08-12'),
    ('Birmingham', 52.4862, -1.8904, '2024-07-31'),
    ('Manchester', 53.4808, -2.2426, '2024-07-19'),
    ('Leeds', 53.8008, -1.5491, '2024-07-19'),
    ('Glasgow', 55.8642, -4.2518, '2024-05-11'),
]

def download(case):
    city, lat, lon, target = case
    target_date = date.fromisoformat(target)
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': (target_date - timedelta(days=1)).isoformat(),
        'end_date': (target_date + timedelta(days=1)).isoformat(),
        'hourly': 'precipitation', 'models': 'era5',
        'timezone': 'GMT', 'timeformat': 'unixtime',
        'precipitation_unit': 'mm', 'cell_selection': 'nearest',
    }
    url = BASE_URL + '?' + urlencode(params)
    with urlopen(Request(url, headers={'User-Agent': 'ResearchDataDownload/1.0'}), timeout=45) as response:
        raw = response.read()
    data = json.loads(raw)
    assert data['utc_offset_seconds'] == 0
    assert data['hourly_units']['precipitation'] == 'mm'
    hourly = data['hourly']
    assert len(hourly['time']) == len(hourly['precipitation'])
    rows = []
    for timestamp, amount in zip(hourly['time'], hourly['precipitation']):
        # Open-Meteo precipitation at t is the sum over the preceding hour, (t-1h, t].
        end_utc = datetime.fromtimestamp(timestamp, timezone.utc)
        start_utc = end_utc - timedelta(hours=1)
        start_local, end_local = start_utc.astimezone(UK), end_utc.astimezone(UK)
        if start_local.date() != target_date:
            continue
        assert amount is not None and isinstance(amount, (int, float)) and amount >= 0
        rows.append({
            'city': city, 'local_date': target,
            'hour_start_local': start_local.isoformat(timespec='minutes'),
            'hour_end_local': end_local.isoformat(timespec='minutes'),
            'hour_start_utc': start_utc.isoformat(timespec='minutes'),
            'hour_end_utc': end_utc.isoformat(timespec='minutes'),
            'precipitation_mm': amount,
            'requested_latitude': lat, 'requested_longitude': lon,
            'returned_grid_latitude': data['latitude'],
            'returned_grid_longitude': data['longitude'],
            'dataset': 'ERA5 via Open-Meteo',
        })
    assert len(rows) == 24, (city, len(rows))
    assert len({r['hour_start_utc'] for r in rows}) == 24
    assert datetime.fromisoformat(rows[0]['hour_start_local']).hour == 0
    assert datetime.fromisoformat(rows[-1]['hour_end_local']).date() == target_date + timedelta(days=1)
    total = sum((Decimal(str(r['precipitation_mm'])) for r in rows), Decimal('0'))
    meta = {
        'city': city, 'peak_date_from_supplementary_table_2': target,
        'requested_coordinates': [lat, lon],
        'returned_grid_coordinates': [data['latitude'], data['longitude']],
        'model': 'ERA5', 'nominal_resolution': '0.25 degrees (~25 km)',
        'timezone_in_output': 'Europe/London, DST-aware conversion from UTC',
        'precipitation_period': 'preceding hour at API timestamp; output labels both start and end',
        'precipitation_unit': 'mm per one-hour interval',
        'records': len(rows), 'selected_local_day_total_mm': float(total),
        'hours_with_precipitation_gt_0_mm': sum(r['precipitation_mm'] > 0 for r in rows),
        'download_url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    return case, raw, rows, meta

results = list(ThreadPoolExecutor(max_workers=5).map(download, CASES))
all_rows, metadata, summaries = [], [], []

def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())

for case, raw, rows, meta in results:
    city, _, _, target = case
    (OUT / 'raw_api' / f'{city}_{target}_ERA5.json').write_bytes(raw)
    write_csv(OUT / f'{city}_{target}_hourly_precipitation.csv', rows)
    all_rows.extend(rows); metadata.append(meta)
    summaries.append({
        'city': city, 'local_date': target,
        'hourly_records': len(rows), 'daily_precipitation_mm': meta['selected_local_day_total_mm'],
        'hours_with_precipitation_gt_0_mm': meta['hours_with_precipitation_gt_0_mm'],
        'requested_latitude': case[1], 'requested_longitude': case[2],
        'returned_grid_latitude': meta['returned_grid_coordinates'][0],
        'returned_grid_longitude': meta['returned_grid_coordinates'][1],
    })

assert len(all_rows) == 120
assert len({(r['city'],r['hour_start_utc']) for r in all_rows}) == 120
write_csv(OUT / 'UK_5Cities_PeakDates_2024_Hourly_Precipitation.csv', all_rows)
write_csv(OUT / 'Daily_summary.csv', summaries)
(OUT / 'Metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'README.txt').write_text('''Hourly precipitation for selected peak-temperature dates in five UK cities

The output contains 24 local-hour records for each selected city-date (120 rows
in total). Data are ERA5 reanalysis grid estimates obtained through the Open-Meteo
Historical Weather API, not rain-gauge observations and not city-boundary-weighted
averages. Requested and returned coordinates are retained in the output metadata.

Open-Meteo reports each precipitation value for the preceding hour. This script
labels both interval boundaries, converts UTC to Europe/London with historical
daylight-saving rules, and retains the original millimetre unit. Hourly values do
not resolve the 11:30 and 19:30 half-hour boundaries. The figure script therefore
reports bounds rather than assuming a within-hour rainfall distribution.

The selected dates are a diagnostic subset and cannot replace a full-year
precipitation control or demonstrate the absence of omitted-variable bias.
''', encoding='utf-8')

for p in OUT.rglob('*'):
    if p.is_file():
        with open(p,'rb') as f: os.fsync(f.fileno())
print(json.dumps({'records':len(all_rows),'summary':summaries,'data_directory':str(OUT)},ensure_ascii=False,indent=2))
