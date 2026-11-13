"""Texas Flood Check: citizen + responder dashboard over the Texas flood photos and the Dell's AI.

Run on the Dell:  streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
"""
import base64
import io
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

APP_NAME = "Texas Flood Check"
DATA = Path(os.environ.get("DATA_DIR", Path.home() / "Texas_Flood_Student_Data"))
SUBMISSIONS = Path(os.environ.get("SUBMISSIONS", Path(__file__).parent / "submissions.csv"))
AI_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")
AI_MODEL = os.environ.get("VLM_MODEL", "/home/dell/models/qwen3-VL-8b-instruct")
ADMIN_CODE = os.environ.get("ADMIN_CODE", "flood2026")
EVENT_COLORS = {"Hurricane Harvey": [37, 99, 235], "July 2025 Texas Hill Country flooding": [234, 88, 12]}
PRIORITY_COLORS = {"High": [220, 38, 38], "Medium": [234, 179, 8], "Low": [34, 197, 94]}

st.set_page_config(page_title=APP_NAME, page_icon="🌊", layout="wide")
st.markdown("""<style>
.hero {padding: 2.2rem 2.4rem; border-radius: 18px; color: white;
       background: linear-gradient(120deg, #0b3d91 0%, #1565c0 55%, #0ea5e9 100%);}
.hero h1 {color: white; font-size: 2.6rem; margin: 0 0 .4rem 0;}
.hero p {font-size: 1.15rem; opacity: .95; margin: 0; max-width: 46rem;}
.card {border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem 1.2rem; height: 100%;}
.card h3 {margin-top: 0;}
.pill {display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-weight: 600; color: white;}
</style>""", unsafe_allow_html=True)


# ---------------- data ----------------
@st.cache_data
def load_photos():
    df = pd.read_csv(DATA / "image_metadata.csv", low_memory=False)
    df["has_file"] = df["image_path"].map(lambda p: (DATA / str(p)).exists())
    return df


@st.cache_data
def load_places():
    df = load_photos()
    # one dot per place: the "after" photo, with its "before" pair if there is one
    after = df[(df["image_role"] == "response") & df["latitude"].notna()].copy()
    after["place"] = after["locality"].fillna("Unknown") + " (" + after["image_id"] + ")"
    after["color"] = after["event_name"].map(EVENT_COLORS).apply(lambda c: c if isinstance(c, list) else [100, 100, 100])
    return after.reset_index(drop=True)


def fact(row, col, fmt="{}", unit=""):
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) == "":
        return "not recorded"
    try:
        return fmt.format(float(v)) + unit if fmt != "{}" else f"{v}{unit}"
    except (ValueError, TypeError):
        return f"{v}{unit}"


def area_facts(row):
    return {
        "Place": fact(row, "locality"),
        "County": fact(row, "county_name"),
        "Flood": fact(row, "event_name"),
        "Photo date": fact(row, "acquisition_date"),
        "Rain that day": fact(row, "precipitation_mm", "{:.0f}", " mm"),
        "Rain in the 7 days before": fact(row, "preceding_7d_precipitation_mm", "{:.0f}", " mm"),
        "Ground elevation": fact(row, "elevation_m", "{:.0f}", " m"),
        "Roads and paths in the photo": fact(row, "osm_road_or_path_count", "{:.0f}"),
        "Nearest high-water mark": fact(row, "nearest_high_water_mark_distance_km", "{:.1f}", " km"),
        "County poverty rate": fact(row, "county_poverty_percent", "{:.1f}", "%"),
        "County median household income": fact(row, "county_median_household_income_usd", "${:,.0f}"),
        "Photo source": fact(row, "source_agency"),
    }


def photo_path(row):
    p = DATA / str(row["image_path"])
    return p if p.exists() else None


# ---------------- AI ----------------
def data_url(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask_ai(messages, max_tokens=450):
    body = {"model": AI_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    req = urllib.request.Request(AI_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer NONE"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"The AI didn't answer ({e}). Try again in a moment."


SYSTEM = ("You help people understand aerial flood photos from Texas. Plain words, short. "
          "Say what you can see and what you cannot tell from a photo taken from above. "
          "A normal river, pond or pool is not a flood. If two photos are given, the first is BEFORE and the second is AFTER.")


# ---------------- pages ----------------
def home():
    places = load_places()
    photos = load_photos()
    st.markdown(f"""<div class="hero"><h1>🌊 {APP_NAME}</h1>
    <p>See what floods did to Texas neighborhoods from the air, ask an AI what it sees,
    and get a plan for your household in five questions. Responders see who needs help first.</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    c[0].metric("Aerial photos", f"{len(photos):,}")
    c[1].metric("Places on the map", f"{len(places):,}")
    c[2].metric("Floods covered", photos["event_name"].nunique())
    c[3].metric("Counties", photos["county_name"].nunique())
    st.write("")
    a, b, d = st.columns(3)
    with a:
        st.markdown('<div class="card"><h3>🗺️ Explore the map</h3>Pick a neighborhood, compare before and after, and ask the AI about it.</div>', unsafe_allow_html=True)
        st.page_link(PAGES["map"], label="Open the map", icon="➡️")
    with b:
        st.markdown('<div class="card"><h3>✅ Check my household</h3>Five questions. You get a plan, and responders know if you need help.</div>', unsafe_allow_html=True)
        st.page_link(PAGES["check"], label="Start the check", icon="➡️")
    with d:
        st.markdown('<div class="card"><h3>📚 Where this comes from</h3>Every photo and number, its source, and what the AI cannot tell.</div>', unsafe_allow_html=True)
        st.page_link(PAGES["sources"], label="See the sources", icon="➡️")
    st.caption("The AI (Qwen3-VL-8B) runs on a Dell Pro Max GB10 at this table. No photo or answer leaves the building.")


def flood_map():
    places = load_places()
    st.title("🗺️ Flood map")
    st.caption("Blue: Hurricane Harvey, Houston, 2017.  Orange: Hill Country flood, July 2025. Click a dot, or pick from the list.")
    event = st.radio("Flood", ["All"] + list(EVENT_COLORS), horizontal=True)
    shown = places if event == "All" else places[places["event_name"] == event]
    layer = pdk.Layer("ScatterplotLayer", shown, id="places", get_position="[longitude, latitude]",
                      get_fill_color="color", get_radius=350, radius_min_pixels=4, pickable=True, auto_highlight=True)
    view = pdk.ViewState(latitude=shown["latitude"].mean(), longitude=shown["longitude"].mean(), zoom=7.2)
    sel = st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light",
                                   tooltip={"text": "{locality}\n{acquisition_date}"}),
                          on_select="rerun", selection_mode="single-object", key="map", height=430)
    picked = (sel.selection.get("objects", {}).get("places") or [None])[0] if sel and sel.selection else None
    names = shown["place"].tolist()
    default = names.index(picked["place"]) if picked and picked.get("place") in names else 0
    choice = st.selectbox("Place", names, index=default)
    row = shown[shown["place"] == choice].iloc[0]
    show_place(row)


def show_place(row):
    photos = load_photos()
    pair = photos[photos["image_id"] == row.get("paired_image_id")]
    frames = ([pair.iloc[0]] if not pair.empty else []) + [row]
    st.subheader(f"{row['locality']}, {row.get('county_name', '')}")
    cols = st.columns(len(frames))
    for c, f in zip(cols, frames):
        p = photo_path(f)
        label = "BEFORE" if f["image_role"] == "before" else "AFTER"
        if p:
            c.image(str(p), caption=f"{label} | {f['acquisition_date']} | {f['source_agency']}", width="stretch")
        else:
            c.info(f"{label} photo isn't on this computer.")
    with st.expander("Details for this place", expanded=True):
        facts = area_facts(row)
        half = len(facts) // 2 + 1
        l, r = st.columns(2)
        for k, v in list(facts.items())[:half]:
            l.markdown(f"**{k}:** {v}")
        for k, v in list(facts.items())[half:]:
            r.markdown(f"**{k}:** {v}")

    key = f"chat-{row['image_id']}"
    st.session_state.setdefault(key, [])
    st.markdown("#### Ask the AI about this place")
    quick = ["Is floodwater covering normally dry land? Give one visible clue.",
             "Which roads look under water, and which direction looks dry enough to leave?",
             "What changed between before and after?" if len(frames) == 2 else "Are any homes surrounded by water?"]
    qc = st.columns(3)
    clicked = next((q for c, q in zip(qc, quick) if c.button(q, key=f"{key}-{q}")), None)
    for m in st.session_state[key]:
        st.chat_message(m["role"]).write(m["text"])
    question = st.chat_input("Ask anything about these photos") or clicked
    if question:
        st.session_state[key].append({"role": "user", "text": question})
        st.chat_message("user").write(question)
        imgs = [photo_path(f) for f in frames if photo_path(f)]
        msgs = [{"role": "system", "content": SYSTEM}]
        for i, m in enumerate(st.session_state[key]):
            if i == 0:
                content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in imgs]
                content.append({"type": "text", "text": m["text"]})
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": m["role"], "content": m["text"]})
        with st.spinner("The AI is looking at the photos..."):
            answer = ask_ai(msgs)
        st.session_state[key].append({"role": "assistant", "text": answer})
        st.chat_message("assistant").write(answer)


def priority(car, mobility, near_water):
    need = (car == "No") + (mobility == "Yes")
    risk = near_water in ("Yes", "Not sure")
    if need and risk:
        return "High"
    if need or risk:
        return "Medium"
    return "Low"


def household_check():
    places = load_places()
    st.title("✅ Check my household")
    st.caption("Five questions. Your answers give you a plan and tell responders if you need help first. Name and phone are optional.")
    with st.form("check"):
        area = st.selectbox("1. Which area do you live closest to?", sorted(places["place"].unique()))
        car = st.radio("2. Do you have a car you can use to leave?", ["Yes", "No"], horizontal=True)
        mobility = st.radio("3. Does anyone at home need help moving (older adult, disability, baby)?", ["No", "Yes"], horizontal=True)
        alerts = st.radio("4. Can you get emergency alerts on a phone?", ["Yes", "No", "Not sure"], horizontal=True)
        near_water = st.radio("5. Is your home near a creek, river, or on low ground?", ["No", "Yes", "Not sure"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("First name (optional)")
        phone = c2.text_input("Phone (optional)")
        lang = c3.selectbox("Plan language", ["English", "Spanish"])
        sent = st.form_submit_button("Get my plan", type="primary")
    if not sent:
        return
    row = places[places["place"] == area].iloc[0]
    level = priority(car, mobility, near_water)
    st.markdown(f'Your priority level: <span class="pill" style="background: rgb{tuple(PRIORITY_COLORS[level])}">{level}</span>',
                unsafe_allow_html=True)
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "phone": phone, "area": area,
              "latitude": row["latitude"], "longitude": row["longitude"], "car": car, "mobility_help": mobility,
              "alerts": alerts, "near_water": near_water, "priority": level, "sample": False}
    pd.DataFrame([record]).to_csv(SUBMISSIONS, mode="a", header=not SUBMISSIONS.exists(), index=False)
    facts = "\n".join(f"- {k}: {v}" for k, v in area_facts(row).items())
    answers = (f"Has a car: {car}. Someone needs help moving: {mobility}. Gets phone alerts: {alerts}. "
               f"Near creek/river/low ground: {near_water}. Priority: {level}.")
    with st.spinner("Writing your plan..."):
        plan = ask_ai([{"role": "system", "content": f"Write in {lang}. Plain words, for a family. Under 170 words. "
                        "Give: 1) why this priority, using only the facts given, 2) three things to do before the next flood, "
                        "3) one line on leaving safely (never drive through water on a road). Don't invent numbers. "
                        "Give county income and poverty only as numbers; never describe people as poor."},
                       {"role": "user", "content": f"Area facts:\n{facts}\n\nHousehold answers: {answers}"}])
    st.markdown("#### Your plan")
    st.write(plan)
    st.success("Saved. If your priority is High, responders will see your household first.")


def sources():
    photos = load_photos()
    st.title("📚 Sources and limits")
    counts = photos.groupby(["event_name", "source_agency"]).size().rename("photos").reset_index()
    st.dataframe(counts, width="stretch", hide_index=True)
    st.markdown("""
**Photos**
- NOAA emergency response aerial photos (after the flood): [storms.ngs.noaa.gov](https://storms.ngs.noaa.gov/)
- USDA NAIP aerial photos (before Harvey, 2016): [USDA NAIP](https://naip-usdaonline.hub.arcgis.com/)
- Prepared for this event by the organizers: [Hugging Face: jennifer-fowler/UTSA-flood-images](https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images)

**Numbers next to each photo** (from the same dataset's `image_metadata.csv`)
- Rainfall: nearest weather station. Elevation and slope: USGS terrain data.
- Roads and paths: OpenStreetMap. High-water marks: USGS.
- County income and poverty: U.S. Census American Community Survey.

**The AI**
- Qwen3-VL-8B-Instruct (reads photos and text), running on the Dell Pro Max GB10 at our table.

**What the AI cannot tell**
- A normal river, pond or pool can look like a flood from above.
- Before and after photos are a year apart, so seasons and cameras differ too.
- Rain nearby is context, not proof of what's in a photo.
- Answers are suggestions to check, not facts. Never drive through water on a road.
""")
    st.caption("Organizers' materials: github.com/jennifer-fowler/GB10competition")


def responders():
    st.title("🚑 Responder dashboard")
    if not st.session_state.get("admin"):
        code = st.text_input("Responder code", type="password")
        if st.button("Open"):
            st.session_state.admin = code == ADMIN_CODE
            st.rerun()
        st.caption("For the demo the code is in the team's notes.")
        return
    places = load_places()
    if st.button("Add 12 sample households (clearly marked as samples)"):
        import random
        random.seed(len(places))
        rows = []
        for i in range(12):
            r = places.sample(1, random_state=i).iloc[0]
            car, mob, near = random.choice(["Yes", "No"]), random.choice(["No", "No", "Yes"]), random.choice(["Yes", "No", "Not sure"])
            rows.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": f"Sample {i + 1}", "phone": "",
                         "area": r["place"], "latitude": r["latitude"], "longitude": r["longitude"], "car": car,
                         "mobility_help": mob, "alerts": random.choice(["Yes", "No"]), "near_water": near,
                         "priority": priority(car, mob, near), "sample": True})
        pd.DataFrame(rows).to_csv(SUBMISSIONS, mode="a", header=not SUBMISSIONS.exists(), index=False)
    if not SUBMISSIONS.exists():
        st.info("No households yet. Fill in the check, or add samples.")
        return
    sub = pd.read_csv(SUBMISSIONS)
    order = {"High": 0, "Medium": 1, "Low": 2}
    sub = sub.sort_values(by="priority", key=lambda s: s.map(order))
    c = st.columns(4)
    c[0].metric("Households", len(sub))
    c[1].metric("High priority", int((sub["priority"] == "High").sum()))
    c[2].metric("No car", int((sub["car"] == "No").sum()))
    c[3].metric("Need help moving", int((sub["mobility_help"] == "Yes").sum()))
    sub["color"] = sub["priority"].map(PRIORITY_COLORS)
    layer = pdk.Layer("ScatterplotLayer", sub, get_position="[longitude, latitude]", get_fill_color="color",
                      get_radius=500, radius_min_pixels=6, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_style="light", tooltip={"text": "{name}: {priority}\n{area}"},
                             initial_view_state=pdk.ViewState(latitude=sub["latitude"].mean(),
                                                              longitude=sub["longitude"].mean(), zoom=7)), height=380)
    st.dataframe(sub.drop(columns=["color"]), width="stretch", hide_index=True)
    st.download_button("Download the list (CSV)", sub.drop(columns=["color"]).to_csv(index=False), "households.csv")
    if st.button("AI: where should we send help first?", type="primary"):
        summary = sub.groupby(["area", "priority"]).size().rename("households").reset_index().to_csv(index=False)
        with st.spinner("Thinking..."):
            st.write(ask_ai([{"role": "system", "content": "You advise flood responders. Plain words, under 150 words. "
                              "Use only the table. Name the areas to visit first and why."},
                             {"role": "user", "content": summary}]))


START = os.environ.get("DT_START", "home")  # which page opens first (used by tests)
PAGES = {
    "home": st.Page(home, title="Home", icon="🏠", default=START == "home"),
    "map": st.Page(flood_map, title="Flood map", icon="🗺️", url_path="map", default=START == "map"),
    "check": st.Page(household_check, title="Household check", icon="✅", url_path="check", default=START == "check"),
    "sources": st.Page(sources, title="Sources", icon="📚", url_path="sources", default=START == "sources"),
    "admin": st.Page(responders, title="Responders", icon="🚑", url_path="responders", default=START == "admin"),
}
st.navigation(list(PAGES.values()), position="top").run()
