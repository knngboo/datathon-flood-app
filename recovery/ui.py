"""Additive Streamlit page; existing photo chat and general exploration stay intact."""
import copy
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from .ai import AssessmentError, GB10Provider
from .data import load_rainfall, load_records
from .demo import demo_entries
from .scoring import DEFAULT_WEIGHTS, LABELS, rank


def render(vlm):
    st.title("Recovery Prioritization")
    st.caption("AI-assisted assessment for human investigation and recovery planning. No resources are allocated automatically.")
    mode = st.radio("Evidence source", ["Synthetic demonstration", "Organizer dataset"], horizontal=True,
                    key="recovery_mode")
    synthetic = mode == "Synthetic demonstration"
    if synthetic:
        st.warning("SYNTHETIC DEMONSTRATION: all observations, rainfall and coordinates are fictional. No images were analyzed.")
        if "recovery_demo" not in st.session_state:
            st.session_state.recovery_demo = demo_entries()
        entries = st.session_state.recovery_demo
    else:
        entries = _real_data(vlm)
    with st.expander("How priorities are calculated", expanded=not entries):
        st.write("This illustrative index supports human review; the weights and thresholds have not been validated as emergency-management policy. "
                 "Visual extent levels 0–3 are normalized to 0–100. Unknown or uncertain evidence is excluded, "
                 "and the remaining weights are renormalized. At least two positive-weight factors, including a visual factor, are required.")
        st.write("Rainfall context uses a complete 14-day window before image acquisition, in millimeters, "
                 "scaled linearly to 100 at 200 mm and capped there. This demonstration scale is not a flood threshold. "
                 "Missing, rejected or trace reports leave that factor unknown. Station context does not prove site rainfall or damage.")
        st.write("Rankings compare individual assessments at verified coordinates, not cities or inferred populations. "
                 "Repeated images of a site remain separate assessments. Different evidence coverage can affect comparability; "
                 "inspect contributions and acquisition dates before acting. Every AI assessment needs human review.")
    st.subheader("Scoring weights")
    columns = st.columns(4)
    weights = {key: column.slider(LABELS[key], 0, 100, value, key="recovery_weight_" + key)
               for column, (key, value) in zip(columns, DEFAULT_WEIGHTS.items())}
    if sum(weights.values()) == 0:
        st.error("Set at least one weight above zero.")
        return
    st.caption("Relative weights are normalized automatically. Coverage shows the share of selected weight supported by known evidence, not AI confidence.")
    if not entries:
        st.info("Load metadata and assess images to populate the table, or select the synthetic demonstration.")
        return
    rows = rank(entries, weights)
    table = []
    for row in rows:
        record = row["record"]
        table.append({"Rank": row["rank"], "Image ID": record["image_id"],
                      "Date": record["acquisition_date"], "Locality (metadata)": record["locality"],
                      "Assessment index": round(row["score"], 2) if row["score"] is not None else None,
                      "Evidence coverage (%)": round(row["coverage"], 1),
                      "Geography": "Verified attribution" if row["geographic_eligible"] else "Unverified / unknown",
                      "Human review": row["review"]["status"], "Provider": row["ai_provenance"]["provider"]})
    st.subheader("Ranked assessments")
    st.caption("Unranked records remain visible. Their index, when available, describes an image and does not establish a location priority.")
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    selection = st.selectbox("Inspect assessment", [row["record"]["image_id"] for row in rows])
    selected = next(row for row in rows if row["record"]["image_id"] == selection)
    record = selected["record"]
    st.subheader(selection)
    st.write(selected["explanation"])
    left, right = st.columns(2)
    with left:
        if record["image_path"] and Path(record["image_path"]).is_file():
            st.image(record["image_path"], caption=f"{record['acquisition_date']} | {record['source_agency']}", use_container_width=True)
        else:
            st.info("No image available. This synthetic scenario uses fixed observations." if synthetic else "Image unavailable.")
        st.write("**Supporting observations**")
        for key, value in selected["assessment"]["factors"].items():
            st.write(f"{LABELS[key]} — {value['status']}: {value['evidence']}")
        for key in ("standing_water", "building_exposure", "potential_infrastructure_damage", "other_conditions"):
            st.write(f"{key.replace('_', ' ').capitalize()}: {selected['assessment'][key]}")
        for uncertainty in selected["assessment"]["uncertainties"]:
            st.warning(uncertainty)
    with right:
        st.dataframe(pd.DataFrame([
            {"Factor": LABELS[key], "Normalized value": selected["factors"][key],
             "Selected weight": weights[key], "Points contributed": selected["contributions"][key]}
            for key in DEFAULT_WEIGHTS]), hide_index=True, use_container_width=True)
        rain = selected["rainfall"]
        st.write(f"**Historical rainfall** · station {rain.get('station') or 'unknown'}")
        st.write(f"Station distance (km): {rain.get('distance_km')}")
        st.write(f"Complete-window total (mm): {rain.get('total_mm')}")
        if rain["days"]:
            st.dataframe(pd.DataFrame(rain["days"]), hide_index=True, use_container_width=True)
        for issue in record["issues"] + rain["issues"]:
            st.caption(issue)
        st.json({"metadata_source": record["source"], "rainfall_source": rain["source"],
                 "AI": selected["ai_provenance"], "location_source": record["location_source"]})
    _review(entries, selected, synthetic)
    # A real map is never rendered with synthetic coordinates.
    if not synthetic:
        points = [{"lat": row["record"]["latitude"], "lon": row["record"]["longitude"]}
                  for row in rows if row["rank"] is not None]
        if points:
            st.subheader("Assessments with verified geographic attribution")
            st.map(pd.DataFrame(points))
    st.download_button("Download assessment evidence (JSON)",
                       json.dumps({"synthetic": synthetic, "weights": weights, "assessments": rows}, indent=2, allow_nan=False),
                       "synthetic-assessments.json" if synthetic else "recovery-assessments.json", "application/json")
    st.caption("Review edits are session-only. Export JSON to retain evidence, weights, provenance and review notes.")


def _real_data(vlm):
    import os
    root = st.text_input("Organizer dataset folder", os.environ.get("DATA_DIR", str(Path.home() / "Texas_Flood_Student_Data")))
    dataset_key = str(Path(root).expanduser().resolve())
    if st.session_state.get("recovery_dataset") != dataset_key:
        st.session_state.recovery_dataset = dataset_key
        st.session_state.recovery_records = []
        st.session_state.recovery_real = []
    if st.button("Load metadata"):
        st.session_state.recovery_records = []
        st.session_state.recovery_real = []
        try:
            records, errors = load_records(root)
            st.session_state.recovery_records = records
            for error in errors:
                st.warning(error)
            st.success(f"Loaded {len(records)} valid image records; {len(errors)} rows excluded.")
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"Could not load metadata: {exc}")
    records = st.session_state.get("recovery_records", [])
    if records:
        st.caption("Select response imagery deliberately; before images are not automatically treated as post-flood evidence.")
        selected_ids = st.multiselect("Images to assess (up to 20 per run)", [r["image_id"] for r in records],
                                     default=[r["image_id"] for r in records if r["image_role"] == "response"][:3])
        if st.button("Assess selected images with GB10", disabled=not selected_ids):
            if len(selected_ids) > 20:
                st.error("Select at most 20 images per run.")
            else:
                provider = GB10Provider(vlm["url"], vlm["model"], vlm["key"])
                entries = st.session_state.recovery_real
                for record in records:
                    if record["image_id"] not in selected_ids:
                        continue
                    # Remove stale successful evidence if reassessment fails.
                    entries[:] = [entry for entry in entries if entry["record"]["image_id"] != record["image_id"]]
                    try:
                        with st.spinner(f"Assessing {record['image_id']}..."):
                            assessment, provenance = provider.assess(record)
                        try:
                            rain = load_rainfall(record, root)
                        except (OSError, ValueError, sqlite3.Error) as exc:
                            rain = {"days": [], "total_mm": None, "station": None, "distance_km": None,
                                    "source": None, "issues": [f"Rainfall rejected: {exc}"]}
                        entries.append({"record": copy.deepcopy(record), "assessment": assessment,
                                        "ai_provenance": provenance, "rainfall": rain,
                                        "review": {"status": "Pending", "reviewer": "", "notes": ""}})
                    except AssessmentError as exc:
                        st.error(f"{record['image_id']}: {exc}")
    return st.session_state.get("recovery_real", [])


def _review(entries, selected, synthetic):
    identifier = selected["record"]["image_id"]
    entry = next(item for item in entries if item["record"]["image_id"] == identifier)
    review = entry["review"]
    with st.form(f"review_{synthetic}_{identifier}"):
        st.subheader("Human review")
        statuses = ["Pending", "Reviewed", "Needs investigation"]
        status = st.selectbox("Review status", statuses, index=statuses.index(review["status"]))
        reviewer = st.text_input("Reviewer", review["reviewer"])
        notes = st.text_area("Evidence and verification notes", review["notes"])
        source = st.text_input("Geographic attribution evidence (required to rank coordinates)", entry["record"]["location_source"])
        verified = st.checkbox("I verified that the supplied coordinates identify this image's location",
                               value=entry["record"]["location_verified"])
        st.caption("Reviews record human judgment; they do not certify AI findings as confirmed damage.")
        if st.form_submit_button("Save review"):
            if (status != "Pending" or verified) and (not reviewer.strip() or not notes.strip()):
                st.error("Provide a reviewer and evidence notes for verification or completed review.")
            elif verified and (not source.strip() or entry["record"]["latitude"] is None):
                st.error("Valid supplied coordinates and geographic evidence are required. Correct missing coordinates in metadata and reload.")
            else:
                entry["review"] = {"status": status, "reviewer": reviewer, "notes": notes}
                entry["record"]["location_source"] = source.strip()
                entry["record"]["location_verified"] = verified
                st.rerun()
