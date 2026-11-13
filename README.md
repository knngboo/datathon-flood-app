# Texas Flood Check

Dell Social Impact AI Datathon, UTSA, November 13, 2026.

Triage analysis in a timely manner, for flood assistance. Residents submit a report with their information. The AI analyzes it and identifies how urgent it is. Each request is pushed to first responders and ranked by triage status. This improves access to assistance, reduces the load on responders, and keeps people accountable.

## What it does

**Get help (residents).** Pick your location (search an address, use your phone's location, or tap a map) and answer six yes, no or not sure questions: in immediate danger, water rising, trapped, anyone injured, can you leave safely, anyone elderly or disabled. You review your answers before sending. You get a case number, your place in line, and a short reply from the AI telling you what to do while you wait.

**Responders.** Every request on a map and in a list, ranked Critical, High, Medium, Low. Each request shows why it's ranked there, the closest aerial photo from a past flood with a dot where the request came from, and a historical flood analysis for the area. Hotspots (two or more open requests within 1.5 km) glow on the map so one crew can reach them together. Requests can be marked on the way or resolved.

**How it works page.** The full ranking rules, every place the AI is used, and every data source.

## Ranking

Plain points, visible to anyone: in immediate danger 40, trapped 30, someone injured 25, water rising 15, can't leave safely 15, someone elderly or disabled 10 (half for "not sure"), 2 per extra person up to 4. Flood history adds at most 8 points, so it never outranks someone in danger right now. A NOAA river gauge in flood nearby adds 2 to 4. Critical: in immediate danger, trapped with water rising, or 65 points and up.

## Where the AI is used

Qwen3-VL-8B-Instruct, a model that reads photos and text, runs on the Dell Pro Max GB10 at our table. It writes each resident's reply, reads the aerial photo of the area for floodwater, summarizes decades of FEMA records into a two-sentence historical insight, and turns the open requests and hotspots into a "who to reach first" plan. Every AI-written part has an AI tag on screen.

## Data

| Data | Source |
|---|---|
| Flood-related disasters by county | FEMA disaster declarations (OpenFEMA) |
| Households that applied for help, by ZIP code | FEMA housing assistance records (OpenFEMA) |
| Flood zone at a spot | FEMA National Flood Hazard Layer |
| River levels right now | NOAA National Water Prediction Service |
| Aerial photos after past floods | NOAA emergency response imagery (Harvey 2017, Hill Country 2025) |
| Aerial photos before Harvey | USDA National Agriculture Imagery Program (2016) |
| Photo collection for this event | Hugging Face dataset jennifer-fowler/UTSA-flood-images |
| Addresses, streets, counties | OpenStreetMap (Photon, Nominatim) |

## Run it

On the GB10, with the model served at `http://localhost:8000/v1` and the photos in `~/Texas_Flood_Student_Data`:

```
python3 -m venv .venv && .venv/bin/pip install starlette uvicorn pandas pillow
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8501
```

Settings: `VLM_BASE_URL`, `VLM_MODEL`, `DATA_DIR`. The app is `server.py` (Python, Starlette) and `static/index.html` (one page, Leaflet maps). Requests are saved one per line in `requests.jsonl`. There is no database; a one-day prototype doesn't need one.

## Limits

This is a student prototype, not a real emergency service. Anyone in danger should call 911. The aerial photos show past floods, not what's happening now.
