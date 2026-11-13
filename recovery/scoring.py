"""Illustrative deterministic index, with evidence coverage separate from severity."""
import math

DEFAULT_WEIGHTS = {"flood_impact": 35, "infrastructure_exposure": 25,
                   "access_disruption": 25, "rainfall_context": 15}
LABELS = {"flood_impact": "Visible flood impact", "infrastructure_exposure": "Infrastructure exposure",
          "access_disruption": "Access disruption", "rainfall_context": "Rainfall context"}


def score(assessment, rainfall, weights):
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Provide exactly the four supported weights")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Weights must be finite, nonnegative, and sum to more than zero")
    factors = {key: None for key in DEFAULT_WEIGHTS}
    for key in factors.keys() - {"rainfall_context"}:
        item = assessment["factors"][key]
        if item["status"] == "observed":
            factors[key] = item["level"] / 3 * 100
    # Explicit demonstration normalization, not a flood threshold or causal estimate.
    total = rainfall.get("total_mm")
    if total is not None:
        if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0:
            raise ValueError("Rainfall total must be nonnegative finite millimeters")
        factors["rainfall_context"] = min(total / 200, 1) * 100
    supported = {k: v for k, v in factors.items() if v is not None and weights[k] > 0}
    denominator = sum(weights[k] for k in supported)
    # A rainfall-only or single-factor result is not a sufficient assessment.
    eligible = len(supported) >= 2 and any(k != "rainfall_context" for k in supported)
    contributions = {k: (v * weights[k] / denominator if eligible and k in supported else None)
                     for k, v in factors.items()}
    value = sum(v for v in contributions.values() if v is not None) if eligible else None
    missing = [LABELS[k] for k, v in factors.items() if v is None]
    explanation = ("Insufficient evidence: at least two positive-weight factors, including visual evidence, are required."
                   if not eligible else
                   "; ".join(f"{LABELS[k]} contributes {contributions[k]:.1f} points" for k in supported))
    if missing:
        explanation += ". Unknown: " + ", ".join(missing) + ". Supported weights are renormalized."
    return {"score": value, "factors": factors, "contributions": contributions,
            "coverage": denominator / sum(weights.values()) * 100,
            "explanation": explanation, "requires_review": True}


def rank(entries, weights):
    """Rank individual geolocated assessments, never aggregate images into a locality."""
    rows = []
    for entry in entries:
        result = score(entry["assessment"], entry["rainfall"], weights)
        record = entry["record"]
        geographic = bool(record["location_verified"] and record["location_source"]
                          and record["latitude"] is not None and record["longitude"] is not None)
        rows.append({**entry, **result, "rank": None, "geographic_eligible": geographic})
    # Never compare real evidence with synthetic fixtures in the same run.
    if len({row["record"]["synthetic"] for row in rows}) > 1:
        raise ValueError("Synthetic and real observations must be ranked separately")
    rows.sort(key=lambda row: (not (row["geographic_eligible"] and row["score"] is not None),
                               -(row["score"] if row["score"] is not None else -1), row["record"]["image_id"]))
    position = 0
    for row in rows:
        if row["geographic_eligible"] and row["score"] is not None:
            position += 1
            row["rank"] = position
    return rows
