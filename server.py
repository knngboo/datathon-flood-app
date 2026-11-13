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
            "before_date": before_date, "before_source": before_src,
            "facts": [
                ["Rain in the 7 days before", num(r.get("preceding_7d_precipitation_mm"), "{:,.0f} mm")],
                ["Ground elevation", num(r.get("elevation_m"), "{:,.0f} m")],
                ["Roads and paths in view", num(r.get("osm_road_or_path_count"), "{:,.0f}")],
                ["Nearest high-water mark", (num(hwm * 1000, "{:,.0f} m away") if not pd.isna(hwm) and hwm < 1 else num(hwm, "{:,.1f} km away"))],
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
    return FileResponse(HERE / "static" / "index.html")


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


app = Starlette(routes=[
    Route("/", index), Route("/api/places", api_places), Route("/api/find", api_find),
    Mount("/static", StaticFiles(directory=HERE / "static")), Route("/img/{image_id}", img),
    Route("/api/ask", api_ask, methods=["POST"]), Route("/api/check", api_check, methods=["POST"]),
    Route("/api/households", api_households), Route("/api/households.csv", api_households_csv),
    Route("/api/samples", api_samples, methods=["POST"]), Route("/api/dispatch", api_dispatch, methods=["POST"]),
])
