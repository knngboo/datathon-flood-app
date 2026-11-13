"""Texas Flood Check: a small web server for the flood map, AI questions, household check and responder view.

Run on the Dell:  .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8501
"""
import base64
import csv
import io
import json
import math
import os
import urllib.parse
import random
import urllib.request
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR", Path.home() / "Texas_Flood_Student_Data"))
HOUSEHOLDS = Path(os.environ.get("SUBMISSIONS", HERE / "households.csv"))
AI_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")
AI_MODEL = os.environ.get("VLM_MODEL", "/home/dell/models/qwen3-VL-8b-instruct")
FIELDS = ["time", "name", "phone", "place_id", "place", "lat", "lon", "car", "mobility_help", "alerts", "near_water", "priority", "sample"]

SYSTEM = ("You help people understand aerial flood photos from Texas. Plain words, short. "
          "Say what you can see and what you cannot tell from a photo taken from above. "
          "A normal river, pond or pool is not a flood. When two photos are given, the first is BEFORE and the second is AFTER. "
          "When asked if there is flooding, start your answer with Yes, No, or Unclear.")


def num(v, fmt):
    try:
        return None if pd.isna(v) else fmt.format(float(v))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def photos():
    df = pd.read_csv(DATA / "image_metadata.csv", low_memory=False)
    return df.set_index("image_id", drop=False)


@lru_cache(maxsize=1)
def places():
    df = photos()
    after = df[(df["image_role"] == "response") & df["latitude"].notna()]
    out = []
    for _, r in after.iterrows():
        pair = r["paired_image_id"] if isinstance(r["paired_image_id"], str) and r["paired_image_id"] in df.index else None
        before_date = df.loc[pair, "acquisition_date"] if pair else None
        before_src = df.loc[pair, "source_agency"] if pair else None
        hwm = r.get("nearest_high_water_mark_distance_km")
        out.append({
            "id": r["image_id"], "before": pair, "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            "place": r["locality"] if isinstance(r["locality"], str) else "Unnamed place",
            "county": r["county_name"] if isinstance(r["county_name"], str) else "",
            "event": "harvey" if "Harvey" in str(r["event_name"]) else "hill",
            "event_name": r["event_name"], "date": r["acquisition_date"], "source": r["source_agency"],
            "bounds": [float(r[k]) for k in ("west", "south", "east", "north")] if all(not pd.isna(r.get(k)) for k in ("west", "south", "east", "north")) else None,
            "before_date": before_date, "before_source": before_src,
            "facts": [
                ["Rain in the week before the photo", num(r.get("preceding_7d_precipitation_mm"), "{:,.0f} mm")],
                ["Height above sea level", num(r.get("elevation_m"), "{:,.0f} m")],
                ["Closest recorded flood line", (num(hwm * 1000, "{:,.0f} m away") if not pd.isna(hwm) and hwm < 1 else num(hwm, "{:,.1f} km away"))],
                ["County poverty rate", num(r.get("county_poverty_percent"), "{:.0f}%")],
                ["County median household income", num(r.get("county_median_household_income_usd"), "${:,.0f}")],
            ],
        })
    return out


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def nearest(lat, lon, n=1):
    return sorted(places(), key=lambda p: km(lat, lon, p["lat"], p["lon"]))[:n]


@lru_cache(maxsize=256)
def geocode(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1, "countrycodes": "us", "viewbox": "-100.5,31.5,-93.5,28.5", "bounded": 0})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.1 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        hits = json.loads(r.read())
    if not hits:
        return None
    h = hits[0]
    return float(h["lat"]), float(h["lon"]), h["display_name"]


def place(pid):
    return next((p for p in places() if p["id"] == pid), None)


# ---------- photos, shrunk and cached ----------
@lru_cache(maxsize=512)
def thumb(image_id, width):
    df = photos()
    if image_id not in df.index:
        return None
    path = DATA / str(df.loc[image_id, "image_path"])
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def data_url(image_id):
    b = thumb(image_id, 1024)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode() if b else None


# ---------- AI ----------
def ask_ai(messages, max_tokens=450):
    body = {"model": AI_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    req = urllib.request.Request(AI_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer NONE"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def priority(car, mobility, near_water):
    need = car == "No" or mobility == "Yes"
    risk = near_water in ("Yes", "Not sure")
    return "High" if need and risk else "Medium" if need or risk else "Low"


def save_household(row):
    new = not HOUSEHOLDS.exists()
    with HOUSEHOLDS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_households():
    if not HOUSEHOLDS.exists():
        return []
    with HOUSEHOLDS.open() as f:
        rows = list(csv.DictReader(f))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(rows, key=lambda r: (order.get(r["priority"], 3), r["time"]))


# ---------- routes ----------
async def index(request):
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


async def api_places(request):
    return JSONResponse(places())


async def img(request):
    w = min(int(request.query_params.get("w", 720)), 1024)
    b = await run_in_threadpool(thumb, request.path_params["image_id"], w)
    if b is None:
        return Response(status_code=404)
    return Response(b, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


async def api_ask(request: Request):
    body = await request.json()
    p = place(body.get("id"))
    if not p:
        return JSONResponse({"error": "That place isn't in the data."}, status_code=404)
    history = body.get("history", [])[-8:]
    question = str(body.get("question", "")).strip()[:500]
    images = [i for i in (p["before"], p["id"]) if i]
    content = [{"type": "image_url", "image_url": {"url": u}} for u in map(data_url, images) if u]
    msgs = [{"role": "system", "content": SYSTEM}]
    first = history[0]["text"] if history else question
    msgs.append({"role": "user", "content": content + [{"type": "text", "text": first}]})
    for m in history[1:] + ([{"role": "user", "text": question}] if history else []):
        msgs.append({"role": m["role"], "content": m["text"]})
    try:
        answer = await run_in_threadpool(ask_ai, msgs)
    except Exception as e:
        return JSONResponse({"error": f"The AI didn't answer ({e}). Try again."}, status_code=502)
    return JSONResponse({"answer": answer})


import re

SUFFIX = r"\b(street|st|drive|dr|road|rd|lane|ln|avenue|ave|boulevard|blvd|court|ct|circle|cir|way|trail|trl|parkway|pkwy)\b\.?"


@lru_cache(maxsize=256)
def nominatim(q, limit=5):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": limit, "countrycodes": "us", "addressdetails": 0})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.2 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=6) as r:
        return [(float(h["lat"]), float(h["lon"]), h["display_name"]) for h in json.loads(r.read())]


def suggest(q):
    import time
    q = re.sub(r"\s+", " ", q.strip())
    num = re.match(r"^(\d+)\s", q)
    words = [w for w in re.sub(SUFFIX, "", re.sub(r"^\d+\s+", "", q), flags=re.I).replace(",", " ").split() if len(w) > 2 and w.lower() not in ("tx", "texas")]
    street = words[0].lower() if words else ""
    tx = q if re.search(r"\b(tx|texas)\b", q, re.I) else q + ", Texas"
    no_suffix = re.sub(r"\s+", " ", re.sub(SUFFIX, "", tx, flags=re.I)).strip(" ,")
    tries = list(dict.fromkeys([tx, no_suffix, re.sub(r"^\d+\s+", "", no_suffix)]))
    found = {}
    for i, t in enumerate(tries):
        if i:
            time.sleep(1)  # Nominatim allows one request a second
        for lat, lon, label in nominatim(t):
            if "Texas" not in label or (street and street not in label.lower()):
                continue
            parts = [x.strip() for x in label.split(",") if x.strip() not in ("United States",)]
            short = ", ".join(parts[:4])
            rank = (2 if num and label.startswith(num.group(1)) else 0) + (1 if street else 0)
            found.setdefault(short, {"label": short.replace(f"{num.group(1)}, ", f"{num.group(1)} ") if num else short, "lat": lat, "lon": lon, "rank": rank})
        if any(v["rank"] >= 2 for v in found.values()):
            break
    out = sorted(found.values(), key=lambda v: -v["rank"])[:5]
    for o in out:
        o.pop("rank")
    return out


async def api_geo(request):
    q = request.query_params.get("q", "").strip()[:120]
    if len(q) < 3:
        return JSONResponse({"error": "Type at least a street, town or ZIP code."}, status_code=400)
    try:
        hits = await run_in_threadpool(suggest, q)
        if not hits:
            hits = await run_in_threadpool(photon, q)
    except Exception as e:
        return JSONResponse({"error": "Address search isn't reachable right now. Use your current location instead."}, status_code=502)
    if not hits:
        return JSONResponse({"error": f"No match for \"{q}\" in Texas. Try the street and town, or a ZIP code."}, status_code=404)
    for h in hits:
        n = nearest(h["lat"], h["lon"])[0]
        h["nearest"] = {"id": n["id"], "place": n["place"], "km": round(km(h["lat"], h["lon"], n["lat"], n["lon"]), 1)}
    return JSONResponse(hits)


@lru_cache(maxsize=512)
def photon(q):
    url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode({"q": q, "limit": 6, "lang": "en", "bbox": "-106.65,25.84,-93.51,36.5"})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.2 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=8) as r:
        feats = json.loads(r.read())["features"]
    out = []
    for f in feats:
        p = f["properties"]
        if p.get("state") != "Texas":
            continue
        line1 = " ".join(x for x in [p.get("housenumber", ""), p.get("street") or p.get("name", "")] if x).strip()
        line2 = ", ".join(x for x in [p.get("city") or p.get("county"), "TX", p.get("postcode")] if x)
        lon, lat = f["geometry"]["coordinates"]
        out.append({"label": f"{line1}, {line2}" if line1 else line2, "lat": lat, "lon": lon})
    return list({o["label"]: o for o in out}.values())[:5]


async def api_suggest(request):
    q = request.query_params.get("q", "").strip()[:120]
    if len(q) < 3:
        return JSONResponse([])
    try:
        return JSONResponse(await run_in_threadpool(photon, q))
    except Exception:
        return JSONResponse([])


@lru_cache(maxsize=512)
def reverse(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 17, "addressdetails": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.2 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        a = json.loads(r.read()).get("address", {})
    line1 = " ".join(x for x in [a.get("house_number", ""), a.get("road", "")] if x).strip()
    line2 = ", ".join(x for x in [a.get("city") or a.get("town") or a.get("village") or a.get("county"), "TX", a.get("postcode")] if x)
    return f"Near {line1}, {line2}" if line1 else f"Near {line2}"


async def api_reverse(request):
    try:
        lat, lon = round(float(request.query_params["lat"]), 5), round(float(request.query_params["lon"]), 5)
        return JSONResponse({"label": await run_in_threadpool(reverse, lat, lon)})
    except Exception:
        return JSONResponse({"label": "Pin on the map"})


async def api_find(request):
    q = request.query_params.get("q", "").strip()[:120]
    if not q:
        return JSONResponse({"error": "Type an address, town or ZIP code."}, status_code=400)
    try:
        hit = await run_in_threadpool(geocode, q if "tx" in q.lower() or "texas" in q.lower() else q + ", Texas")
    except Exception as e:
        return JSONResponse({"error": f"Address search isn't reachable right now ({e}). Click a dot on the map instead."}, status_code=502)
    if not hit:
        return JSONResponse({"error": f"Couldn't find \"{q}\". Try a town name or ZIP code."}, status_code=404)
    lat, lon, label = hit
    close = [{"id": p["id"], "place": p["place"], "km": round(km(lat, lon, p["lat"], p["lon"]), 1)} for p in nearest(lat, lon, 3)]
    return JSONResponse({"lat": lat, "lon": lon, "label": label, "nearest": close})


async def api_check(request: Request):
    b = await request.json()
    p = place(b.get("place_id"))
    if not p:
        return JSONResponse({"error": "Pick the area you live closest to."}, status_code=400)
    level = priority(b.get("car"), b.get("mobility_help"), b.get("near_water"))
    save_household({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": str(b.get("name", ""))[:60],
                    "phone": str(b.get("phone", ""))[:30], "place_id": p["id"], "place": p["place"], "lat": p["lat"],
                    "lon": p["lon"], "car": b.get("car"), "mobility_help": b.get("mobility_help"), "alerts": b.get("alerts"),
                    "near_water": b.get("near_water"), "priority": level, "sample": "no"})
    facts = "\n".join(f"- {k}: {v}" for k, v in [["Place", f"{p['place']}, {p['county']}"], ["Flood", p["event_name"]]] + p["facts"] if v)
    answers = (f"Has a car: {b.get('car')}. Someone needs help moving: {b.get('mobility_help')}. Gets phone alerts: {b.get('alerts')}. "
               f"Near creek/river/low ground: {b.get('near_water')}. Priority: {level}.")
    lang = "Spanish" if b.get("lang") == "Spanish" else "English"
    try:
        plan = await run_in_threadpool(ask_ai, [
            {"role": "system", "content": f"Write in {lang}. Plain words, for a family. Under 160 words. Give: 1) why this priority, "
             "using only the facts given, 2) three things to do before the next flood, 3) one line on leaving safely "
             "(never drive through water on a road). Don't invent numbers. Give county income and poverty only as numbers; "
             "never describe people as poor. No headings."},
            {"role": "user", "content": f"Area facts:\n{facts}\n\nHousehold answers: {answers}\n\nWrite the whole plan in {lang}."}])
    except Exception as e:
        plan = f"Your answers were saved, but the AI didn't answer ({e})."
    return JSONResponse({"priority": level, "plan": plan})


async def api_households(request):
    return JSONResponse(read_households())


async def api_households_csv(request):
    if not HOUSEHOLDS.exists():
        return PlainTextResponse(",".join(FIELDS) + "\n")
    return Response(HOUSEHOLDS.read_text(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=households.csv"})


async def api_samples(request):
    rng = random.Random()
    for i in range(12):
        p = rng.choice(places())
        car, mob, near = rng.choice(["Yes", "Yes", "No"]), rng.choice(["No", "No", "Yes"]), rng.choice(["Yes", "No", "Not sure"])
        save_household({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": f"Sample household {i + 1}", "phone": "",
                        "place_id": p["id"], "place": p["place"], "lat": p["lat"], "lon": p["lon"], "car": car,
                        "mobility_help": mob, "alerts": rng.choice(["Yes", "No"]), "near_water": near,
                        "priority": priority(car, mob, near), "sample": "yes"})
    return JSONResponse({"ok": True})


async def api_dispatch(request):
    rows = read_households()
    if not rows:
        return JSONResponse({"answer": "No households yet."})
    table = pd.DataFrame(rows).groupby(["place", "priority"]).size().rename("households").reset_index().to_csv(index=False)
    try:
        answer = await run_in_threadpool(ask_ai, [
            {"role": "system", "content": "You advise flood responders. Plain words, under 130 words, no headings. "
             "Use only the table. Name the areas to visit first and say why in one line each."},
            {"role": "user", "content": table}], 350)
    except Exception as e:
        return JSONResponse({"error": f"The AI didn't answer ({e})."}, status_code=502)
    return JSONResponse({"answer": answer})


REQUESTS = Path(os.environ.get("REQUESTS", HERE / "requests.jsonl"))
import threading as _t
LOCK = _t.Lock()
LEVELS = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
DEPTH = {"ankle": ("Water ankle deep", 10), "knee": ("Water knee deep", 20), "waist": ("Water waist deep or higher", 35), "unsure": ("Depth not sure", 10)}
NEED = {"stay": "Needs a place to stay", "cleanup": "Needs help cleaning out", "repairs": "Needs repairs",
        "supplies": "Needs food, water or medicine", "none": "Nothing needed right now"}
HELP = {
    "rent": ["FEMA rental assistance, if you can't live in your home right now",
             "FEMA help replacing essential belongings (furniture, appliances, clothes)",
             "A renters insurance claim, if your policy covers flooding",
             "Ask your landlord in writing about repairs and your lease"],
    "own": ["FEMA home repair assistance, to make your home safe to live in",
            "A flood insurance claim, if you have flood insurance (file quickly, with photos)",
            "FEMA temporary housing help while repairs happen",
            "SBA disaster loans for repairs insurance doesn't cover"],
}
APPLY = "Apply at DisasterAssistance.gov or call FEMA at 1-800-621-3362. Take photos of all damage before you clean up."
SAMPLE_PHOTOS = []


FLOOD_TYPES = ("Flood", "Hurricane", "Severe Storm", "Severe Storm(s)", "Coastal Storm", "Tropical Storm")


@lru_cache(maxsize=256)
def where_is(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 16, "addressdetails": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.3 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=8) as r:
        a = json.loads(r.read()).get("address", {})
    return (a.get("county") or "").replace(" County", ""), (a.get("postcode") or "")[:5], a.get("state", "")


@lru_cache(maxsize=128)
def county_declarations(county):
    url = ("https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?" + urllib.parse.urlencode({
        "$filter": f"state eq 'TX' and designatedArea eq '{county} (County)'",
        "$select": "disasterNumber,declarationTitle,incidentType,declarationDate,declarationType", "$orderby": "declarationDate desc", "$top": 500}))
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.3"}), timeout=20) as r:
        rows = json.loads(r.read())["DisasterDeclarationsSummaries"]
    seen = {}
    for x in rows:
        if x["incidentType"] in FLOOD_TYPES:
            seen.setdefault(x["disasterNumber"], x)
    return [{"number": x["disasterNumber"], "year": int(x["declarationDate"][:4]), "title": x["declarationTitle"].title(),
             "type": x["incidentType"], "major": x["declarationType"] == "DR"} for x in seen.values()]


CACHE_FILE = HERE / "cache.json"
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {"history": {}, "verdicts": {}}


def save_cache():
    CACHE_FILE.write_text(json.dumps(CACHE))


def flood_history(lat, lon):
    key = f"{round(lat, 3)},{round(lon, 3)}"
    if key in CACHE["history"] and "classes" in (CACHE["history"][key] or {"classes": 1}):
        return CACHE["history"][key]
    h = _flood_history(lat, lon)
    CACHE["history"][key] = h
    save_cache()
    return h


def _flood_history(lat, lon):
    county, zipcode, state = where_is(round(lat, 3), round(lon, 3))
    if state != "Texas" or not county:
        return None
    events = county_declarations(county)
    if not events:
        return {"county": county, "zip": zipcode, "total": 0, "events": []}
    this_year = datetime.now().year
    decades = {}
    for e in events:
        d = f"{e['year'] // 10 * 10}s"; decades[d] = decades.get(d, 0) + 1
    severity = []
    graded = []
    for e in events:
        n = None
        if zipcode and e["major"] and e["year"] >= 2004 and len([g for g in graded if g["households"] is not None]) < 12:
            try:
                f = fema(zipcode, e["number"])
                n = f["renters"]["applied"] + f["owners"]["applied"]
            except Exception:
                n = None
        cls = "Minor" if not e["major"] else "Severe" if (n or 0) >= 5000 else "Major" if (n or 0) >= 100 else "Moderate"
        graded.append({**e, "households": n, "class": cls})
        if n:
            severity.append({"year": e["year"], "title": e["title"], "households": n})
    classes = {c: sum(1 for g in graded if g["class"] == c) for c in ("Minor", "Moderate", "Major", "Severe")}
    return {"county": county, "zip": zipcode, "total": len(events), "since": min(e["year"] for e in events),
            "last10": sum(1 for e in events if e["year"] >= this_year - 10), "last_year": max(e["year"] for e in events),
            "decades": dict(sorted(decades.items())), "severity": severity[:6], "classes": classes,
            "major_count": classes["Major"] + classes["Severe"], "events": graded}


def history_points(h):
    if not h or not h.get("total"):
        return 0, None
    pts = 3 if h["last10"] >= 5 else 2 if h["last10"] >= 2 else 1
    worst = max((x["households"] for x in h["severity"]), default=0)
    pts += 3 if worst >= 1000 else 1 if worst >= 100 else 0
    return pts, f"Flood-prone area: {h['last10']} flood disasters in {h['county']} County in 10 years"


@lru_cache(maxsize=512)
def flood_zone(lat, lon):
    url = ("https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query?" + urllib.parse.urlencode({
        "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF", "returnGeometry": "false", "f": "json"}, safe=","))
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.3"}), timeout=12) as r:
        feats = json.loads(r.read()).get("features", [])
    if not feats:
        return None
    a = feats[0]["attributes"]
    high = a.get("SFHA_TF") == "T"
    return {"zone": a.get("FLD_ZONE"), "high_risk": high,
            "label": "High-risk flood zone (1% chance of flooding every year)" if high else "Outside FEMA's high-risk flood zone"}


GAUGES = {}


def live_gauges(lat, lon):
    import time
    key = f"{round(lat, 2)},{round(lon, 2)}"
    if key in GAUGES and time.time() - GAUGES[key][0] < 300:
        return GAUGES[key][1]
    url = ("https://api.water.noaa.gov/nwps/v1/gauges?" + urllib.parse.urlencode({
        "bbox.xmin": lon - .15, "bbox.ymin": lat - .15, "bbox.xmax": lon + .15, "bbox.ymax": lat + .15, "srid": "EPSG_4326"}))
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.3"}), timeout=12) as r:
        gs = json.loads(r.read()).get("gauges", [])
    out = []
    for g in gs:
        obs = (g.get("status") or {}).get("observed") or {}
        if g.get("latitude") is None:
            continue
        out.append({"name": g.get("name"), "id": g.get("lid"), "km": round(km(lat, lon, g["latitude"], g["longitude"]), 1),
                    "category": obs.get("floodCategory") or "unknown", "stage": obs.get("primary"), "unit": obs.get("primaryUnit") or "ft"})
    out = sorted(out, key=lambda x: x["km"])[:3]
    GAUGES[key] = (time.time(), out)
    return out


HISTORY_SUMMARY = {}


async def api_history(request):
    try:
        lat, lon = float(request.query_params["lat"]), float(request.query_params["lon"])
        h = await run_in_threadpool(flood_history, lat, lon)
    except Exception as e:
        return JSONResponse({"error": f"FEMA's records didn't load ({e})."}, status_code=502)
    if not h:
        return JSONResponse({"error": "No FEMA flood history for this spot (outside Texas or no county found)."}, status_code=404)
    try:
        h = {**h, "zone": await run_in_threadpool(flood_zone, round(lat, 4), round(lon, 4))}
    except Exception:
        h = {**h, "zone": None}
    try:
        h["gauges"] = await run_in_threadpool(live_gauges, lat, lon)
    except Exception:
        h["gauges"] = []
    key = (h["county"], h["zip"])
    if key not in HISTORY_SUMMARY:
        facts = (f"{h['county']} County, Texas: {h['total']} federal flood-related disasters since {h.get('since')} "
                 f"({h['classes']['Minor']} minor, {h['classes']['Moderate']} moderate, {h['classes']['Major']} major, {h['classes']['Severe']} severe), "
                 f"{h.get('last10')} in the last 10 years, most recent {h.get('last_year')}. Households in ZIP {h['zip']} that asked FEMA for help: "
                 + ("; ".join(f"{x['title']} {x['year']}: {x['households']:,}" for x in h["severity"]) or "no records") + ".")
        try:
            HISTORY_SUMMARY[key] = await run_in_threadpool(ask_ai, [
                {"role": "system", "content": "You brief flood rescue crews. Exactly two short plain sentences, under 45 words total. First: how often and how badly this area has flooded. Second: what that history suggests for a rescue happening now (for example deep water, flooded streets, many people needing help at once). Never tell responders to prioritize people because they applied to FEMA in the past. Use only the numbers given. The household numbers are households that asked FEMA for help: say exactly that, never 'displaced', 'affected' or 'damaged'."},
                {"role": "user", "content": facts}], 120)
        except Exception:
            HISTORY_SUMMARY[key] = ""
    return JSONResponse({**h, "summary": HISTORY_SUMMARY[key]})


def history_near(lat, lon):
    p = nearest(lat, lon)[0]
    return {"id": p["id"], "place": p["place"], "event": p["event_name"], "event_key": p["event"],
            "km": round(km(lat, lon, p["lat"], p["lon"]), 2)}


VERDICTS = {k: tuple(v) for k, v in CACHE["verdicts"].items()}


def photo_verdict(pid):
    if pid in VERDICTS:
        return VERDICTS[pid]
    p = place(pid)
    images = [i for i in (p["before"], p["id"]) if i]
    content = [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
    q = (("Compare the BEFORE photo (first) and the AFTER photo (second). " if p["before"] else
          "This aerial photo was taken days after a flood. ") +
         "Is floodwater or fresh flood damage visible on homes or streets? Start with Yes, No, or Unclear, then one short sentence naming the visible clue.")
    text = ask_ai([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content + [{"type": "text", "text": q}]}], 120).strip()
    w = re.split(r"\W+", text.lower(), maxsplit=1)[0] if text else ""
    verdict = "Yes" if w == "yes" else "No" if w == "no" else "Unclear"
    rest = re.sub(r"^\W*(yes|no|unclear)\W*", "", text, flags=re.I)
    VERDICTS[pid] = (verdict, rest[:1].upper() + rest[1:])
    CACHE["verdicts"][pid] = list(VERDICTS[pid])
    save_cache()
    return VERDICTS[pid]


EMERGENCY = {"danger": ("In immediate danger", 40, 20), "trapped": ("Trapped", 30, 15), "injured": ("Someone injured", 25, 12),
             "rising": ("Water rising", 15, 8), "vulnerable": ("Elderly or disabled person", 10, 5)}


def score(a, people, hist_pts, hist_why, verdict):
    pts, why = 0, []
    for k, (label, yes, unsure) in EMERGENCY.items():
        if a.get(k) == "yes":
            pts += yes; why.append(label)
        elif a.get(k) == "unsure":
            pts += unsure; why.append(label + " (not sure)")
    if a.get("evacuate") == "no":
        pts += 15; why.append("Can't leave safely")
    elif a.get("evacuate") == "unsure":
        pts += 8; why.append("Not sure they can leave safely")
    pts += min(max(people - 1, 0), 4) * 2
    if hist_pts:
        pts += hist_pts; why.append(hist_why)
    if verdict == "Yes":
        pts += 2; why.append("Aerial photo shows this area flooded before")
    if a.get("danger") == "yes" or (a.get("trapped") == "yes" and a.get("rising") == "yes") or pts >= 65:
        level = "Critical"
    else:
        level = "High" if pts >= 35 else "Medium" if pts >= 15 else "Low"
    return pts, level, why


def read_requests():
    if not REQUESTS.exists():
        return []
    rows = [json.loads(l) for l in REQUESTS.read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: (r["status"] == "Resolved", LEVELS.get(r["level"], 9), -r["score"], r["time"]))


def write_requests(rows):
    REQUESTS.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_report(lat, lon, label, a, people, name, phone, sample=False, hist_at=None):
    hist = history_near(lat, lon)
    verdict, vtext = None, ""
    if sample:
        verdict, vtext = VERDICTS.get(hist["id"], (None, ""))
    elif hist["km"] <= 3:
        try:
            verdict, vtext = photo_verdict(hist["id"])
        except Exception:
            verdict, vtext = None, ""
    try:
        fh = flood_history(*(hist_at or (lat, lon)))
    except Exception:
        fh = None
    hist_pts, hist_why = history_points(fh) if fh else (0, None)
    try:
        g = live_gauges(*(hist_at or (lat, lon)))
    except Exception:
        g = []
    flooding = [x for x in g if x["category"] in ("minor", "moderate", "major")]
    if flooding:
        hist_pts += {"minor": 2, "moderate": 3, "major": 4}[flooding[0]["category"]]
        hist_why = (hist_why + "; " if hist_why else "") + f"Live river gauge in {flooding[0]['category']} flood (NOAA)"
    pts, level, why = score(a, people, hist_pts, hist_why, verdict)
    return {"time": datetime.now().strftime("%H:%M"), "lat": lat, "lon": lon, "label": label, "answers": a, "people": people,
            "name": name, "phone": phone, "score": pts, "level": level, "why": why, "history": hist,
            "verdict": verdict, "verdict_text": vtext, "hist_pts": hist_pts, "status": "New", "sample": sample}


def write_reply(r, case_id, updated):
    facts = (f"Case number {case_id}{' (updated an earlier request)' if updated else ''}. Priority {r['level']}. "
             f"What they told us: {', '.join(r['why']) or 'nothing urgent'}. People with them: {r['people']}.")
    try:
        return ask_ai([
            {"role": "system", "content": "You reply to someone in Texas who just asked for help during a flash flood. Calm, plain words, "
             "3 or 4 short sentences, no lists, no headings. Use only the facts given; never promise when help arrives. Mention their case number. "
             "Tell them exactly what to do right now for their situation: stay out of moving water, never drive through flooded roads, "
             "go to the highest safe place (not a closed attic), keep the phone charged. If they are in immediate danger, tell them to call 911 now."},
            {"role": "user", "content": facts}], 200)
    except Exception:
        return (f"Your request is in, case {case_id}. Stay out of moving water and never drive through a flooded road. Get to the highest safe "
                "place you can, keep your phone charged, and call 911 if you are in immediate danger.")


async def api_request(request: Request):
    import time
    b = await request.json()
    try:
        lat, lon = float(b["lat"]), float(b["lon"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"error": "We need your location first."}, status_code=400)
    a = {k: str(b.get("answers", {}).get(k, "")) for k in ("danger", "rising", "trapped", "injured", "evacuate", "vulnerable")}
    if not all(a.values()):
        return JSONResponse({"error": "Some questions weren't answered. Go back and answer them."}, status_code=400)
    people = max(1, min(int(b.get("people") or 1), 50))
    r = await run_in_threadpool(make_report, lat, lon, str(b.get("label", ""))[:160], a, people,
                                str(b.get("name", ""))[:60], str(b.get("phone", ""))[:30])
    r["ts"] = time.time()
    with LOCK:
        r = save_request(r, lat, lon)
    dup = r.pop("_dup")
    pos = [x["id"] for x in read_requests() if x["status"] != "Resolved"].index(r["id"]) + 1
    reply = await run_in_threadpool(write_reply, r, r["id"], bool(dup))
    return JSONResponse({"id": r["id"], "updated": bool(dup), "level": r["level"], "position": pos, "reply": reply})


def save_request(r, lat, lon):
    import time
    rows = read_requests()
    dup = next((x for x in rows if x["status"] != "Resolved" and not x.get("sample") and time.time() - x.get("ts", 0) < 1800
                and km(lat, lon, x["lat"], x["lon"]) < 0.05), None)
    if dup:
        r["id"] = dup["id"]
        rows = [r if x["id"] == dup["id"] else x for x in rows]
    else:
        r["id"] = f"R{len(rows) + 1:03d}"
        rows.append(r)
    write_requests(rows)
    r["_dup"] = bool(dup)
    return r


async def api_requests(request):
    return JSONResponse(read_requests())


async def api_status(request: Request):
    b = await request.json()
    with LOCK:
        rows = read_requests()
        for r in rows:
            if r["id"] == request.path_params["rid"] and b.get("status") in ("New", "On the way", "Resolved"):
                r["status"] = b["status"]
        write_requests(rows)
    return JSONResponse({"ok": True})


SAMPLES = [
    ({"danger": "yes", "rising": "yes", "trapped": "yes", "injured": "no", "evacuate": "no", "vulnerable": "yes"}, 3),
    ({"danger": "no", "rising": "yes", "trapped": "no", "injured": "yes", "evacuate": "unsure", "vulnerable": "no"}, 2),
    ({"danger": "unsure", "rising": "yes", "trapped": "no", "injured": "no", "evacuate": "no", "vulnerable": "yes"}, 4),
    ({"danger": "no", "rising": "no", "trapped": "no", "injured": "no", "evacuate": "yes", "vulnerable": "no"}, 1),
    ({"danger": "no", "rising": "unsure", "trapped": "yes", "injured": "no", "evacuate": "no", "vulnerable": "no"}, 2),
    ({"danger": "no", "rising": "no", "trapped": "no", "injured": "no", "evacuate": "unsure", "vulnerable": "yes"}, 2),
    ({"danger": "no", "rising": "yes", "trapped": "yes", "injured": "no", "evacuate": "no", "vulnerable": "no"}, 2),
    ({"danger": "unsure", "rising": "yes", "trapped": "no", "injured": "yes", "evacuate": "no", "vulnerable": "no"}, 5),
]


def sample_photo_ids():
    if not SAMPLE_PHOTOS:
        seen = set()
        for p in places():
            if p["place"] not in seen and p["before"]:
                seen.add(p["place"]); SAMPLE_PHOTOS.append(p["id"])
    return SAMPLE_PHOTOS[:6]


def warm_samples():
    import time
    for pid in sample_photo_ids():
        p = place(pid)
        key = f"{round(p['lat'], 3)},{round(p['lon'], 3)}"
        try:
            if key not in CACHE["history"]:
                flood_history(p["lat"], p["lon"]); time.sleep(1)
        except Exception:
            pass
    for pid in sample_photo_ids():
        try:
            photo_verdict(pid)
        except Exception:
            pass
    if not REQUESTS.exists():
        seed_samples()


def seed_samples(reset=False):
    with LOCK:
        _seed(reset)


def _seed(reset):
    rows = [] if reset else read_requests()
    rng = random.Random()
    ids = sample_photo_ids()
    ids = ids + [ids[0], ids[0]]  # two more near the first spot, so the demo has a real hotspot
    for i, ((a, people), pid) in enumerate(zip(SAMPLES, ids)):
        p = place(pid)
        lat, lon = p["lat"] + rng.uniform(-.006, .006), p["lon"] + rng.uniform(-.006, .006)
        r = make_report(lat, lon, f"Near {p['place']}", a, people, "", "", True, (p["lat"], p["lon"]))
        r["id"] = f"R{len(rows) + 1:03d}"
        rows.append(r)
    write_requests(rows)


async def api_samples2(request):
    await run_in_threadpool(seed_samples, True)
    return JSONResponse({"ok": True})


async def api_briefing(request):
    rows = [r for r in read_requests() if r["status"] != "Resolved"]
    if not rows:
        return JSONResponse({"answer": "No open reports."})
    table = "\n".join(f"{r['id']} | {r['level']} | {r['label']} | {', '.join(r['why'])} | people {r['people']} | {r['status']}" for r in rows[:25])
    seen, groups = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        g = [r]; seen.add(r["id"])
        for x in g:
            for o in rows:
                if o["id"] not in seen and km(x["lat"], x["lon"], o["lat"], o["lon"]) <= 1.5:
                    seen.add(o["id"]); g.append(o)
        if len(g) >= 2:
            groups.append(", ".join(x["id"] for x in g))
    if groups:
        table += "\n\nHotspots (requests within 1.5 km of each other, one crew can reach them together): " + "; ".join(groups)
    try:
        answer = await run_in_threadpool(ask_ai, [
            {"role": "system", "content": "You brief flood rescue crews. Plain words, under 110 words, no headings. "
             "Say which requests to reach first and why, by request ID. If a hotspot contains urgent requests, say one crew can cover that hotspot together. Use only the list."},
            {"role": "user", "content": table}], 300)
    except Exception:
        return JSONResponse({"error": "The AI didn't answer. Try again."}, status_code=502)
    return JSONResponse({"answer": answer})


# ================= neighborhood app =================
DISASTER = {"harvey": (4332, "Hurricane Harvey"), "hill": (4879, "July 2025 Hill Country flood")}


@lru_cache(maxsize=1)
def hoods():
    groups = {}
    for p in places():
        g = groups.setdefault(p["place"], {"name": p["place"], "event": p["event"], "event_name": p["event_name"], "counties": set(), "photos": []})
        if p["county"]:
            g["counties"].add(p["county"])
        g["photos"].append(p)
    out = []
    for g in groups.values():
        ph = sorted(g["photos"], key=lambda p: (p["before"] is None, p["id"]))
        out.append({"name": g["name"], "event": g["event"], "event_name": g["event_name"], "counties": sorted(g["counties"]),
                    "lat": sum(p["lat"] for p in ph) / len(ph), "lon": sum(p["lon"] for p in ph) / len(ph),
                    "count": len(ph), "cover": ph[0]["id"],
                    "photos": [{k: p[k] for k in ("id", "before", "date", "before_date", "source", "before_source", "lat", "lon", "facts")} for p in ph]})
    return sorted(out, key=lambda h: (h["event"] != "harvey", -h["count"]))


async def api_hoods(request):
    return JSONResponse([{k: v for k, v in h.items() if k != "photos"} for h in hoods()])


async def api_hood(request):
    h = next((x for x in hoods() if x["name"] == request.query_params.get("name")), None)
    return JSONResponse(h) if h else JSONResponse({"error": "We don't have photos for that neighborhood."}, status_code=404)


@lru_cache(maxsize=256)
def zip_at(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 16, "addressdetails": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.2 (UTSA datathon demo)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return (json.loads(r.read()).get("address", {}).get("postcode") or "")[:5]


@lru_cache(maxsize=256)
def fema(zipcode, disaster):
    out = {}
    for kind in ("Renters", "Owners"):
        url = (f"https://www.fema.gov/api/open/v2/HousingAssistance{kind}?" +
               urllib.parse.urlencode({"$filter": f"disasterNumber eq {disaster} and zipCode eq '{zipcode}'", "$top": 100}))
        req = urllib.request.Request(url, headers={"User-Agent": "TexasFloodCheck/0.2"})
        with urllib.request.urlopen(req, timeout=20) as r:
            rows = json.loads(r.read()).get(f"HousingAssistance{kind}", [])
        out[kind.lower()] = {"applied": sum(x.get("validRegistrations") or 0 for x in rows),
                             "inspected": sum(x.get("totalInspected") or 0 for x in rows),
                             "approved_people": sum(x.get("approvedForFemaAssistance") or 0 for x in rows),
                             "approved_dollars": round(sum(x.get("totalApprovedIhpAmount") or 0 for x in rows))}
    return out


async def api_fema(request):
    q = request.query_params
    event = q.get("event", "harvey")
    disaster, title = DISASTER.get(event, DISASTER["harvey"])
    zipcode = (q.get("zip") or "").strip()[:5]
    try:
        if not zipcode:
            zipcode = await run_in_threadpool(zip_at, round(float(q["lat"]), 4), round(float(q["lon"]), 4))
        if not zipcode.isdigit():
            return JSONResponse({"error": "Couldn't work out the ZIP code for this spot."}, status_code=404)
        data = await run_in_threadpool(fema, zipcode, disaster)
    except Exception as e:
        return JSONResponse({"error": f"FEMA's data service didn't answer ({e})."}, status_code=502)
    return JSONResponse({"zip": zipcode, "disaster": disaster, "title": title, **data})


async def api_ready(request: Request):
    b = await request.json()
    tenure = "rents their home" if b.get("tenure") == "rent" else "owns their home"
    facts = (f"Neighborhood: {b.get('hood')}. Flood: {b.get('event_name')}. The person {tenure}. "
             f"What the aerial photo showed: {str(b.get('photo_summary', 'not checked'))[:400]}. "
             f"FEMA records for ZIP {b.get('zip')}: {b.get('fema_line', 'not available')}.")
    try:
        tips = await run_in_threadpool(ask_ai, [
            {"role": "system", "content": "You help Texas residents prepare for the next flood. Plain words. Give exactly 4 short tips "
             "as lines starting with '- ', specific to the facts given (renters vs owners matters: renters need renters insurance "
             "with flood coverage and their landlord's plan; owners need flood insurance, which usually takes 30 days to start). "
             "No headings, nothing else."},
            {"role": "user", "content": facts}], 260)
    except Exception:
        return JSONResponse({"error": "The AI didn't answer. Try again."}, status_code=502)
    return JSONResponse({"tips": [t.strip()[2:].strip() for t in tips.splitlines() if t.strip().startswith("-")][:4] or [tips]})


FEEDBACK = HERE / "feedback.jsonl"


async def api_feedback(request: Request):
    if request.method == "POST":
        b = await request.json()
        text = str(b.get("text", "")).strip()[:1000]
        if not text:
            return JSONResponse({"error": "Write what's wrong first."}, status_code=400)
        row = {"time": datetime.now().strftime("%H:%M"), "name": str(b.get("name", "")).strip()[:40] or "Someone",
               "where": str(b.get("where", ""))[:200], "text": text}
        with FEEDBACK.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return JSONResponse({"ok": True})
    rows = [json.loads(l) for l in FEEDBACK.read_text().splitlines() if l.strip()] if FEEDBACK.exists() else []
    return JSONResponse(rows[::-1])


async def home(request):
    return FileResponse(HERE / "static" / "home.html")


import threading
threading.Thread(target=warm_samples, daemon=True).start()

app = Starlette(routes=[
    Route("/", index), Route("/api/history", api_history), Route("/api/feedback", api_feedback, methods=["GET", "POST"]), Route("/api/hoods", api_hoods), Route("/api/hood", api_hood),
    Route("/api/fema", api_fema), Route("/api/ready", api_ready, methods=["POST"]), Route("/api/places", api_places), Route("/api/find", api_find), Route("/api/geo", api_geo), Route("/api/suggest", api_suggest), Route("/api/reverse", api_reverse),
    Route("/api/request", api_request, methods=["POST"]), Route("/api/requests", api_requests),
    Route("/api/requests/{rid}/status", api_status, methods=["POST"]), Route("/api/requests/samples", api_samples2, methods=["POST"]),
    Route("/api/requests/briefing", api_briefing, methods=["POST"]),
    Mount("/static", StaticFiles(directory=HERE / "static")), Route("/img/{image_id}", img),
    Route("/api/ask", api_ask, methods=["POST"]), Route("/api/check", api_check, methods=["POST"]),
    Route("/api/households", api_households), Route("/api/households.csv", api_households_csv),
    Route("/api/samples", api_samples, methods=["POST"]), Route("/api/dispatch", api_dispatch, methods=["POST"]),
])
