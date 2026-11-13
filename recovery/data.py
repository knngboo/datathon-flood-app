"""Validated, provenance-preserving adapters for organizer metadata and rainfall."""
import csv
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path


def number(value, low, high):
    try:
        result = float(value)
        return result if math.isfinite(result) and low <= result <= high else None
    except (ValueError, TypeError):
        return None


def text(value):
    return str(value).strip() if value is not None else ""


def validate_record(raw, root, source):
    root = Path(root).resolve()
    issues = []
    identifier = text(raw.get("image_id"))
    if not identifier:
        raise ValueError("Missing image_id")
    relative = text(raw.get("image_path") or raw.get("filename"))
    image = (root / relative).resolve() if relative else None
    if image and not image.is_relative_to(root):
        raise ValueError(f"{identifier}: image path escapes dataset folder")
    if not image or not image.is_file():
        issues.append("Image unavailable")
        image = None
    latitude = number(raw.get("latitude"), -90, 90)
    longitude = number(raw.get("longitude"), -180, 180)
    if latitude is None or longitude is None:
        latitude = longitude = None
        issues.append("Coordinates missing or invalid")
    location_source = text(raw.get("location_source"))
    verified = (raw.get("location_verified") is True or text(raw.get("location_verified")).lower() == "true")
    verified = bool(verified and location_source and latitude is not None)
    if not verified:
        issues.append("Geographic attribution requires human verification")
    acquisition = text(raw.get("acquisition_date"))
    try:
        acquisition = date.fromisoformat(acquisition).isoformat()
    except ValueError:
        acquisition = None
        issues.append("Acquisition date missing or invalid")
    return {
        "image_id": identifier, "image_path": str(image) if image else None,
        "acquisition_date": acquisition, "locality": text(raw.get("locality")),
        "event_name": text(raw.get("event_name")), "image_role": text(raw.get("image_role")),
        "latitude": latitude, "longitude": longitude,
        "location_source": location_source, "location_verified": verified,
        "source": str(source), "source_agency": text(raw.get("source_agency")),
        "synthetic": False, "issues": issues, "metadata": dict(raw),
    }


def load_records(root):
    """Read student CSV or annotation manifest; reject duplicate IDs rather than join ambiguously."""
    root = Path(root).expanduser().resolve()
    source = root / "image_metadata.csv"
    if source.is_file():
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        source = root / "manifest.json"
        rows = json.loads(source.read_text())
        if not isinstance(rows, list):
            raise ValueError("manifest.json must contain an array of image records")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Metadata rows must be objects")
    counts = Counter(text(row.get("image_id")) for row in rows)
    records, errors = [], []
    for index, row in enumerate(rows, 1):
        if counts[text(row.get("image_id"))] > 1:
            errors.append(f"Row {index}: duplicate image_id; excluded")
            continue
        try:
            records.append(validate_record(row, root, source))
        except ValueError as exc:
            errors.append(f"Row {index}: {exc}")
    return records, errors


def load_rainfall(record, root, days=14):
    """Use explicit station/window keys. No spatial or date-only guesses.

    PRCP is millimeters in the competition export. Preserve missing/rejected days;
    do not score partial windows or trace amounts as exact totals.
    """
    meta = record["metadata"]
    station = text(meta.get("precipitation_station_id"))
    window = text(meta.get("weather_window_id"))
    empty = {"days": [], "total_mm": None, "station": station,
             "distance_km": number(meta.get("precipitation_station_distance_km"), 0, 20000),
             "source": None, "issues": []}
    if not record["acquisition_date"] or not station or not window:
        empty["issues"].append("Rainfall unavailable: date, station or weather window missing")
        return empty
    end = date.fromisoformat(record["acquisition_date"])
    start = end - timedelta(days=days)
    root = Path(root).expanduser().resolve()
    database, csv_path = root / "observations.sqlite", root / "daily_weather.csv"
    if database.is_file():
        empty["source"] = str(database)
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute(
                'SELECT DATE, PRCP, PRCP_ATTRIBUTES FROM weather_daily_weather '
                'WHERE window_id = ? AND STATION = ? AND DATE >= ? AND DATE < ?',
                (window, station, start.isoformat(), end.isoformat()))]
    elif csv_path.is_file():
        empty["source"] = str(csv_path)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"window_id", "STATION", "DATE", "PRCP", "PRCP_ATTRIBUTES"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("daily_weather.csv missing required columns")
            rows = [row for row in reader if row["window_id"] == window and row["STATION"] == station
                    and start.isoformat() <= row["DATE"] < end.isoformat()]
    else:
        empty["issues"].append("Rainfall file unavailable")
        return empty
    by_day = {}
    for row in rows:
        day = row["DATE"]
        if day in by_day:
            raise ValueError(f"Duplicate station/date rainfall: {day}")
        flags = text(row["PRCP_ATTRIBUTES"]).split(",")
        rejected = flags[0] in {"P", "I"} or (len(flags) > 1 and bool(flags[1]))
        mm = None if rejected else number(row["PRCP"], 0, 10000)
        by_day[day] = {"rain_mm": mm, "trace": flags[0] == "T" and mm is not None}
    for offset in range(days):
        day = (start + timedelta(days=offset)).isoformat()
        empty["days"].append({"date": day, **by_day.get(day, {"rain_mm": None, "trace": False})})
    if all(row["rain_mm"] is not None and not row["trace"] for row in empty["days"]):
        empty["total_mm"] = sum(row["rain_mm"] for row in empty["days"])
    else:
        empty["issues"].append("Incomplete or trace rainfall window: context factor unknown")
    return empty
