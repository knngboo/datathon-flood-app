# Datathon flood app

Team project for the Dell Social Impact AI Datathon (UTSA, Nov 13, 2026).

The AI (Qwen3-VL-8B, a model that reads photos) runs on the team's Dell GB10. This app sends flood photos and questions to it and shows the answers.

## Run it on the Dell
```
python -m streamlit run app.py --server.address 0.0.0.0
```
Then open `http://<dell address>:8501`. The AI must be running at `http://localhost:8000/v1`.

`DECISIONS.md` has what we decided and why.

## Recovery Prioritization (adam branch)

The new page extends `app.py`; the existing photo, map and chat features remain available.
All work is local. No deployment, push or merge is required.

### Install, run and test

Python 3.10+ is required. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m unittest discover -s tests -v
```

Open http://localhost:8501 and choose **Recovery Prioritization** in the sidebar.
Choose **Synthetic demonstration** to run without a GB10 or datasets. Every demo
record, coordinate and rainfall value is explicitly synthetic; no photo inference
occurs. Increase visible flood impact to 100 and reduce the other weights to 1 to
see SYN-A move above SYN-B. SYN-C has missing factors and unverified geography;
SYN-D has insufficient evidence. Synthetic coordinates are never put on the map.

### Use organizer data and GB10

Set `DATA_DIR` to the organizer's `Texas_Flood_Student_Data` directory, or enter it
on the page. Configure the existing sidebar vision server controls, or set
`VLM_BASE_URL` (ending in `/v1`), `VLM_MODEL`, and `VLM_API_KEY`. These settings are
shared with the original flood-photo page. Do not commit credentials. The
server must already be running and reachable from this machine.

Choose **Organizer dataset**, load metadata, select up to 20 images and click
**Assess selected images with GB10**. Errors are shown per image; unsuccessful
reassessments remove old results rather than displaying stale evidence. There is
no automatic mock fallback. Results, reviews and settings are session-only;
download JSON to retain evidence and provenance. Exported image paths refer to
the original local files and do not bundle photos. There is no export reimport yet.

Accepted image metadata:

| Field | Handling |
| --- | --- |
| `image_id` | Required unique ID; all occurrences of duplicate IDs are excluded. |
| `image_path` or `filename` | Relative path inside dataset folder; absent images remain flagged. |
| `acquisition_date` | ISO `YYYY-MM-DD`; missing/invalid dates disable rainfall association. |
| `event_name`, `locality`, `image_role`, `source_agency` | Preserved descriptive metadata; locality alone is not location evidence. |
| `latitude`, `longitude` | Finite WGS84 degrees in valid ranges; never inferred from an image or name. |
| `location_verified`, `location_source` | Optional explicit `true` and geographic evidence. Coordinates alone do not enable ranking. Human reviewers may confirm supplied coordinates with their name and notes. |
| `weather_window_id`, `precipitation_station_id` | Both required for exact rainfall association. |
| `precipitation_station_distance_km` | Optional station context; does not imply rain at the photographed site. |

The loader accepts `image_metadata.csv`, or a JSON-array `manifest.json` with the
same fields (`filename` is accepted). It retains raw metadata and source paths.
It does not perform uncertain filename, locality or demographic joins. Organizer
annotation manifests without verified coordinates remain image assessments.

Rainfall is read from `observations.sqlite`, table `weather_daily_weather`, or
`daily_weather.csv` in the selected folder. Expected fields: `window_id`,
`STATION`, `DATE`, `PRCP`, `PRCP_ATTRIBUTES`. `PRCP` must be in **millimeters**, as
in the competition export; convert other products before importing. The adapter
matches station and window, checks the 14 days preceding acquisition, rejects
duplicate dates, negative/nonfinite values, failed quality checks and presumed
or unreported zeros (`P`/`I`). Missing or trace reports leave the total unknown.
SQLite is opened read-only. Historical rainfall is context, not proof of damage.

### Index and human review

Initial relative weights are flood impact 35, infrastructure exposure 25, access
disruption 25 and rainfall context 15. Visual extent levels 0–3 map to 0–100;
uncertain factors stay unknown. Complete rainfall totals map linearly to 0–100
at 200 mm, capped at 100. These are **illustrative, unvalidated choices**, not
emergency-management policy, hazard thresholds or measured damage estimates.

The index is `sum(normalized_factor * weight) / sum(supported_weights)`.
At least two positive-weight factors, including visual evidence, are required.
Unknown factors are excluded, not set to zero; evidence coverage separately
reports the share of selected weights with support. Indices with different
coverage or dates may not be directly comparable. A low score does not establish
safety. An observed zero describes only the visible area in that image.

Only sufficiently supported assessments with verified geographic attribution
receive a location rank or map point. Images are ranked individually, never
aggregated into a city or site; repeated images can describe the same location.
Ties break by image ID. The page shows normalized factors, contributions,
supporting observations, uncertainty, source information and human review notes.
Every AI result requires human review; a review label does not confirm damage.
Schema checks cannot guarantee that a model's visual interpretation is correct.
No water depth, economic loss or population estimate is calculated.

### Modules and team integration

- `recovery/data.py`: validation, provenance, image metadata and rainfall adapters.
- `recovery/ai.py`: constrained JSON schema, GB10 provider and synthetic-only mock.
- `recovery/scoring.py`: pure deterministic scoring and geographic ranking.
- `recovery/ui.py`: Streamlit page, review controls and export.
- `recovery/demo.py`: fixed synthetic scenarios.
- `tests/`: unit tests and Streamlit headless integration tests.

The main shared-file integration point is `app.py` navigation. Small existing-page
guards handle absent datasets without crashing and avoid scanning the whole home
directory when no photo folder is installed. `chat.py` is unchanged. No model,
organizer dataset, coordinates, demographic data or verified damage measurements
are bundled. Live GB10 accuracy and performance still require testing on the
team's server with actual organizer imagery.

Reference schemas and server behavior were inspected in the competition's
[README](https://github.com/jennifer-fowler/GB10competition) and
[texas_flood.py](https://github.com/jennifer-fowler/GB10competition/blob/main/texas_flood.py).
This feature uses its own adapters and the existing app's urllib request pattern;
the external helper is not vendored.

### Verification in this checkout

All 30 unit and Streamlit interaction tests passed using the existing Anaconda
Python 3.13 / Streamlit 1.45.1 environment. The preview on port 8502 returned
`ok` from `/_stcore/health`. Full installation into the new `.venv` was not
completed because dependency downloads stalled and the direct PyPI retry could
not resolve its host. Use the installation steps above on a working connection
for the complete dependency set, including PyDeck for maps. Browser visual QA
was unavailable because the browser access host could not be reached. No live
GB10 inference or real-data map was verified.
