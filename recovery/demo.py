"""Fixed synthetic scenarios: no actual images, locations or rainfall measurements."""
from .ai import MockProvider


def demo_entries():
    entries = []
    for identifier, levels, rain, coordinates in [
        ("SYN-A", (3, 1, 1), 40, (29.45, -98.50)),
        ("SYN-B", (1, 3, 3), 80, (29.46, -98.51)),
        ("SYN-C", (2, None, 2), None, None),
        ("SYN-D", (None, None, None), None, None),
    ]:
        factors = {}
        for key, level in zip(("flood_impact", "infrastructure_exposure", "access_disruption"), levels):
            factors[key] = {"status": "observed" if level is not None else "unknown", "level": level,
                            "evidence": (f"Synthetic scenario specifies visible extent level {level}."
                                         if level is not None else "Synthetic scenario supplies no evidence.")}
        record = {
            "image_id": identifier, "image_path": None, "acquisition_date": "2026-01-15",
            "locality": "Fictional demonstration site", "event_name": "Synthetic event", "image_role": "response",
            "latitude": coordinates[0] if coordinates else None,
            "longitude": coordinates[1] if coordinates else None,
            "location_verified": bool(coordinates), "location_source": "Synthetic coordinate fixture" if coordinates else "",
            "synthetic": True, "source": "recovery/demo.py (synthetic)", "source_agency": "None",
            "metadata": {}, "issues": ["Synthetic example; no actual image or verified real-world condition"],
            "mock_assessment": {"factors": factors, "standing_water": "Synthetic standing-water scenario.",
                                "building_exposure": "No actual building observations.",
                                "potential_infrastructure_damage": "No damage established.",
                                "other_conditions": "No actual image was analyzed.",
                                "uncertainties": ["All evidence and coordinates are invented for demonstration."]},
        }
        assessment, provenance = MockProvider().assess(record)
        entries.append({"record": record, "assessment": assessment, "ai_provenance": provenance,
                        "review": {"status": "Pending", "reviewer": "", "notes": ""},
                        "rainfall": {"days": [], "total_mm": rain, "station": "SYNTHETIC",
                                     "source": "Synthetic fixture", "distance_km": None, "issues": []}})
    return entries
