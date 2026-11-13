import copy
import csv
import json
import sqlite3
import tempfile
import unittest
import urllib.error
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from recovery.ai import AssessmentError, GB10Provider, MockProvider, parse_response
from recovery.data import load_rainfall, load_records, validate_record
from recovery.demo import demo_entries
from recovery.scoring import DEFAULT_WEIGHTS, rank, score


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.entries = demo_entries()

    def metadata(self, **values):
        return {"image_id": "test", "acquisition_date": "2026-01-15", **values}

    def record(self, **values):
        return validate_record(self.metadata(**values), self.root, "test.csv")

    def write_csv(self, filename, rows):
        with (self.root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_coordinates_unknown(self):
        record = self.record(latitude="NaN", longitude="-98")
        self.assertIsNone(record["latitude"])
        self.assertIsNone(record["longitude"])
        self.assertFalse(record["location_verified"])

    def test_geography_requires_evidence(self):
        record = self.record(latitude=29, longitude=-98, location_verified="true")
        self.assertFalse(record["location_verified"])
        record = self.record(latitude=29, longitude=-98, location_verified="true", location_source="Verified geotag")
        self.assertTrue(record["location_verified"])

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.record(image_path="../private.jpg")

    def test_missing_image_date_and_id(self):
        record = self.record(acquisition_date="not a date")
        self.assertIsNone(record["acquisition_date"])
        self.assertIsNone(record["image_path"])
        with self.assertRaises(ValueError):
            self.record(image_id="")

    def test_duplicate_ids_excluded(self):
        self.write_csv("image_metadata.csv", [self.metadata(), self.metadata()])
        records, errors = load_records(self.root)
        self.assertEqual(records, [])
        self.assertEqual(len(errors), 2)

    def test_csv_and_manifest_keep_provenance(self):
        self.write_csv("image_metadata.csv", [self.metadata()])
        records, errors = load_records(self.root)
        self.assertFalse(errors)
        self.assertTrue(records[0]["source"].endswith("image_metadata.csv"))
        (self.root / "image_metadata.csv").unlink()
        (self.root / "manifest.json").write_text(json.dumps([self.metadata()]))
        self.assertTrue(load_records(self.root)[0][0]["source"].endswith("manifest.json"))

    def test_rain_requires_exact_keys(self):
        self.assertIsNone(load_rainfall(self.record(), self.root)["total_mm"])

    def weather_rows(self):
        return [{"window_id": "w", "STATION": "s", "DATE": f"2026-01-{day:02d}",
                 "PRCP": "10", "PRCP_ATTRIBUTES": ",,"} for day in range(1, 15)]

    def rainfall(self):
        return load_rainfall(self.record(weather_window_id="w", precipitation_station_id="s"), self.root)

    def test_complete_rainfall_and_sqlite_adapter(self):
        rows = self.weather_rows()
        self.write_csv("daily_weather.csv", rows)
        result = self.rainfall()
        self.assertEqual(result["total_mm"], 140)
        with closing(sqlite3.connect(self.root / "observations.sqlite")) as connection:
            connection.execute("CREATE TABLE weather_daily_weather (window_id, STATION, DATE, PRCP, PRCP_ATTRIBUTES)")
            connection.executemany("INSERT INTO weather_daily_weather VALUES (?, ?, ?, ?, ?)",
                                   [tuple(row.values()) for row in rows])
            connection.commit()
        self.assertEqual(self.rainfall()["days"], result["days"])

    def test_quality_missing_trace_and_presumed_zero_unknown(self):
        for flag in ["P,,", "I,,", ",X,", "T,,"]:
            rows = self.weather_rows()
            rows[0]["PRCP_ATTRIBUTES"] = flag
            self.write_csv("daily_weather.csv", rows)
            self.assertIsNone(self.rainfall()["total_mm"], flag)
        self.write_csv("daily_weather.csv", self.weather_rows()[1:])
        self.assertIsNone(self.rainfall()["total_mm"])
        self.assertIsNone(self.rainfall()["days"][0]["rain_mm"])

    def test_rainfall_duplicate_rejected(self):
        rows = self.weather_rows()
        self.write_csv("daily_weather.csv", rows + [rows[0]])
        with self.assertRaises(ValueError):
            self.rainfall()

    def test_valid_json_and_fenced_json(self):
        payload = self.entries[0]["assessment"]
        self.assertEqual(parse_response(json.dumps(payload)), payload)
        self.assertEqual(parse_response("```json\n" + json.dumps(payload) + "\n```"), payload)

    def test_invalid_ai_responses_rejected(self):
        for response in ["", "yes", "{}", "[]", None]:
            with self.assertRaises(AssessmentError):
                parse_response(response)
        payload = copy.deepcopy(self.entries[0]["assessment"])
        for level in [-1, 4, True, "2", 1.2]:
            payload["factors"]["flood_impact"]["level"] = level
            with self.assertRaises(AssessmentError):
                parse_response(json.dumps(payload))

    def test_uncertain_factor_cannot_have_level(self):
        payload = copy.deepcopy(self.entries[0]["assessment"])
        payload["factors"]["flood_impact"]["status"] = "uncertain"
        with self.assertRaises(AssessmentError):
            parse_response(json.dumps(payload))

    def test_unsupported_ai_fields_rejected(self):
        payload = copy.deepcopy(self.entries[0]["assessment"])
        payload["water_depth"] = 5
        with self.assertRaises(AssessmentError):
            parse_response(json.dumps(payload))

    def test_weighted_score_is_deterministic_and_explained(self):
        entry = self.entries[0]
        result = score(entry["assessment"], entry["rainfall"], DEFAULT_WEIGHTS)
        self.assertAlmostEqual(result["score"], 35 + 25/3 + 25/3 + 3)
        self.assertAlmostEqual(sum(result["contributions"].values()), result["score"])
        self.assertEqual(result, score(entry["assessment"], entry["rainfall"], DEFAULT_WEIGHTS))
        self.assertIn("contributes", result["explanation"])

    def test_missing_factors_renormalized_not_zero(self):
        entry = self.entries[2]
        result = score(entry["assessment"], entry["rainfall"], DEFAULT_WEIGHTS)
        self.assertAlmostEqual(result["score"], 200/3)
        self.assertEqual(result["coverage"], 60)
        self.assertIsNone(result["contributions"]["rainfall_context"])
        self.assertIsNone(result["factors"]["infrastructure_exposure"])

    def test_insufficient_evidence_unscored(self):
        entry = self.entries[3]
        self.assertIsNone(score(entry["assessment"], entry["rainfall"], DEFAULT_WEIGHTS)["score"])
        self.assertIsNone(score(entry["assessment"], {"total_mm": 100}, DEFAULT_WEIGHTS)["score"])

    def test_weight_change_changes_ranking(self):
        default = rank(self.entries, DEFAULT_WEIGHTS)
        flood_only_emphasis = {**DEFAULT_WEIGHTS, "flood_impact": 100,
                               "infrastructure_exposure": 1, "access_disruption": 1, "rainfall_context": 0}
        changed = rank(self.entries, flood_only_emphasis)
        self.assertEqual(default[0]["record"]["image_id"], "SYN-B")
        self.assertEqual(changed[0]["record"]["image_id"], "SYN-A")
        self.assertTrue(all(row["rank"] is None for row in default if not row["geographic_eligible"]))

    def test_invalid_weights_and_mixed_data(self):
        entry = self.entries[0]
        for weights in [{k: 0 for k in DEFAULT_WEIGHTS}, {**DEFAULT_WEIGHTS, "flood_impact": -1},
                        {**DEFAULT_WEIGHTS, "flood_impact": float("nan")}, {}]:
            with self.assertRaises(ValueError):
                score(entry["assessment"], entry["rainfall"], weights)
        self.entries[0]["record"]["synthetic"] = False
        with self.assertRaises(ValueError):
            rank(self.entries, DEFAULT_WEIGHTS)

    def test_mock_refuses_real_data(self):
        with self.assertRaises(AssessmentError):
            MockProvider().assess(self.record())

    def test_gb10_unavailable_does_not_fall_back(self):
        from PIL import Image
        image = self.root / "test.png"
        Image.new("RGB", (8, 8)).save(image)
        record = self.record(image_path="test.png")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("unavailable")):
            with self.assertRaisesRegex(AssessmentError, "no mock data was substituted"):
                GB10Provider("http://localhost/v1", "Qwen").assess(record)

    def test_gb10_success_uses_image_and_structured_parser(self):
        from PIL import Image
        Image.new("RGB", (8, 8)).save(self.root / "test.png")
        response = {"choices": [{"message": {"content": json.dumps(self.entries[0]["assessment"])}}]}
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            payload, provenance = GB10Provider("http://localhost/v1", "Qwen").assess(self.record(image_path="test.png"))
            sent = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(sent["model"], "Qwen")
            self.assertEqual(sent["messages"][1]["content"][0]["type"], "image_url")
            self.assertEqual(payload, self.entries[0]["assessment"])
            self.assertEqual(provenance["provider"], "GB10")

    def test_gb10_malformed_envelope_rejected(self):
        from PIL import Image
        Image.new("RGB", (8, 8)).save(self.root / "test.png")
        for body in [b"not json", b'{"choices": []}', b'{"choices":[{"message":{"content":"{}"}}]}']:
            with patch("urllib.request.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = body
                with self.assertRaises(AssessmentError):
                    GB10Provider("http://localhost/v1", "Qwen").assess(self.record(image_path="test.png"))

    def test_geographic_verification_controls_rank(self):
        self.entries[0]["record"]["location_verified"] = False
        rows = rank(self.entries, DEFAULT_WEIGHTS)
        first = next(row for row in rows if row["record"]["image_id"] == "SYN-A")
        self.assertIsNone(first["rank"])
        self.assertIsNotNone(first["score"])
        self.entries[0]["record"]["location_verified"] = True
        rows = rank(self.entries, DEFAULT_WEIGHTS)
        self.assertIsNotNone(next(row for row in rows if row["record"]["image_id"] == "SYN-A")["rank"])


if __name__ == "__main__":
    unittest.main()
