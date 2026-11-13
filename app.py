"""Datathon kit: look at any city data, map it, and have the local AI explain it.

Start:  streamlit run app.py --server.address 0.0.0.0
AI:     Ollama on this machine (OLLAMA_URL, default http://localhost:11434), model DT_MODEL (default gpt-oss:20b)
"""
import base64
import io
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("DT_MODEL", "gpt-oss:20b")
# The event's vision AI (GPUStack on the GB10, OpenAI-style). Set in the sidebar.
VLM = {"url": os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1"),
       "model": os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
       "key": os.environ.get("VLM_API_KEY", "")}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

st.set_page_config(page_title="Datathon kit", layout="wide")


# ---------- AI ----------
def ask_ai(messages, think="low"):
    body = {"model": MODEL, "messages": messages, "stream": False, "think": think}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"]
    except Exception as e:
        return f"AI not reachable at {OLLAMA} with model {MODEL}: {e}"


def ask_vlm(question, image_path=None, max_tokens=300):
    content = [{"type": "text", "text": question}]
    if image_path:
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((1280, 1280))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    body = {"model": VLM["model"], "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(VLM["url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {VLM['key'] or 'NONE'}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        return f"ERROR {e.code}: {e.read()[:200].decode(errors='ignore')}"
    except Exception as e:
        return f"ERROR: {e}"


def ai_status():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
        return MODEL in names, names
    except Exception:
        return False, []


# ---------- data ----------
@st.cache_data
def load(path):
    p = Path(path)
    if p.suffix == ".geojson":
        gj = json.loads(p.read_text())
        return pd.DataFrame([f["properties"] for f in gj["features"]]), gj
    return pd.read_csv(p, low_memory=False), None


def color_scale(values):
    v = pd.to_numeric(values, errors="coerce")
    lo, hi = v.quantile(0.02), v.quantile(0.98)
    t = ((v - lo) / (hi - lo if hi > lo else 1)).clip(0, 1).fillna(-1)
    # light yellow -> dark red; grey for no data
    return [[200, 200, 200, 80] if x < 0 else
            [int(255 - 75 * x), int(237 - 200 * x), int(160 - 140 * x), 170] for x in t]


def polygon_map(gj, df, metric, tooltip_cols):
    gj = json.loads(json.dumps(gj))
    cols = color_scale(df[metric])
    for f, c, (_, row) in zip(gj["features"], cols, df.iterrows()):
        f["properties"] = {k: (None if pd.isna(row[k]) else row[k]) for k in tooltip_cols + [metric]}
        f["properties"]["_fill"] = c
    layer = pdk.Layer("GeoJsonLayer", gj, pickable=True, stroked=True, filled=True,
                      get_fill_color="properties._fill", get_line_color=[80, 80, 80, 120],
                      line_width_min_pixels=0.5)
    tip = "<br/>".join(f"<b>{c}</b>: {{{c}}}" for c in tooltip_cols + [metric])
    view = pdk.ViewState(latitude=29.45, longitude=-98.5, zoom=9.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": tip}))


def lat_lon_cols(df):
    low = {c.lower(): c for c in df.columns}
    for a, b in [("lat", "lon"), ("latitude", "longitude"), ("lat", "lng"), ("y", "x")]:
        if a in low and b in low:
            return low[a], low[b]
    return None


# ---------- pages ----------
page = st.sidebar.radio("Page", ["Flood photos", "Explore any data", "311 by neighborhood", "AI helper"])
with st.sidebar.expander("Vision AI server (from the organizers)", expanded=not VLM["key"]):
    VLM["url"] = st.text_input("Server URL (ends in /v1)", VLM["url"])
    VLM["model"] = st.text_input("Model", VLM["model"])
    VLM["key"] = st.text_input("Key", VLM["key"], type="password")
ok, names = ai_status()
st.sidebar.write(("AI ready: " if ok else "AI not ready: ") + MODEL)
if not ok and names:
    st.sidebar.write("Models found: " + ", ".join(names))

if page == "Flood photos":
    st.title("Ask the vision AI about many photos at once")
    guess = next((str(d) for d in [Path.home() / "Texas_Flood_Student_Data", Path(__file__).parent.parent / "Texas_Flood_Student_Data",
                                   Path.home() / "GB10competition" / "datasets"] if d.exists()), str(Path.home()))
    folder = Path(st.text_input("Folder with the photos (looks inside subfolders too)", guess)).expanduser()
    imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT and ".ipynb_checkpoints" not in str(p)) if folder.is_dir() else []
    st.write(f"**{len(imgs)} photos found**")
    meta = None
    metas = sorted(folder.rglob("*metadata*.csv")) if folder.is_dir() else []
    if metas:
        meta = pd.read_csv(metas[0])
        with st.expander(f"Photo details from {metas[0].name}: {len(meta)} rows"):
            st.dataframe(meta.head(200), width="stretch")
    q = st.text_area("Question for every photo",
                     "Is floodwater visibly covering normally dry land? Answer yes, no, or uncertain, then give one visible clue.")
    n = st.slider("How many photos (start small, each takes a few seconds)", 1, max(1, len(imgs)), min(10, max(1, len(imgs))))
    if st.button("Run") and imgs:
        rows, bar = [], st.progress(0.0)
        for i, img in enumerate(imgs[:n]):
            a = ask_vlm(q, img)
            rows.append({"file": img.name, "stem": img.stem, "answer": a,
                         "label": a.split()[0].strip(".,:;*").lower() if a else "", "path": str(img)})
            bar.progress((i + 1) / n)
        st.session_state.flood = pd.DataFrame(rows)
    if "flood" in st.session_state:
        res = st.session_state.flood
        if meta is not None:
            for c in meta.columns:
                vals = meta[c].astype(str)
                key = "file" if vals.isin(res["file"]).any() else "stem" if vals.isin(res["stem"]).any() else None
                if key is None:
                    vals = vals.map(lambda v: Path(v).name)
                    key = "file" if vals.isin(res["file"]).any() else None
                if key:
                    res = res.merge(meta.assign(_k=vals), left_on=key, right_on="_k", how="left").drop(columns="_k")
                    break
        st.bar_chart(res["label"].value_counts())
        st.dataframe(res.drop(columns=["path"]), width="stretch")
        st.download_button("Download the answers (CSV)", res.drop(columns=["path"]).to_csv(index=False), "answers.csv")
        if (ll := lat_lon_cols(res)) is not None:
            pts = res[[ll[0], ll[1], "label"]].dropna().rename(columns={ll[0]: "lat", ll[1]: "lon"})
            pts["color"] = pts["label"].map({"yes": "#d7301f", "no": "#2b8cbe"}).fillna("#999999")
            st.map(pts, color="color", size=60)
        cols = st.columns(4)
        for i, r in res.head(24).iterrows():
            cols[i % 4].image(r["path"], caption=f"{r['file']}: {r['answer'][:120]}")
        if st.button("Have the AI sum up these answers"):
            st.write(ask_vlm("Sum up these photo answers for a city emergency manager in under 150 words. "
                             "Use only what is here.\n\n" + res[["file", "answer"]].to_csv(index=False), max_tokens=400))

elif page == "Explore any data":
    st.title("Explore any data")
    files = sorted(p.name for p in DATA.glob("*") if p.suffix in (".csv", ".geojson"))
    up = st.file_uploader("Drop in the file they hand out (CSV)", type=["csv"])
    if up is not None:
        (DATA / up.name).write_bytes(up.getvalue())
        files = sorted(set(files) | {up.name})
    name = st.selectbox("File", files, index=files.index(up.name) if up is not None else 0)
    df, gj = load(str(DATA / name))
    st.write(f"**{len(df):,} rows, {len(df.columns)} columns**")
    st.dataframe(df.head(200), width="stretch")

    with st.expander("Every column: type, how many blank, example"):
        st.dataframe(pd.DataFrame({
            "type": df.dtypes.astype(str), "blank": df.isna().sum(),
            "different values": df.nunique(), "example": df.iloc[0].astype(str) if len(df) else None,
        }), width="stretch")

    c1, c2 = st.columns(2)
    group = c1.selectbox("Count rows by", ["(none)"] + list(df.columns))
    num = c2.selectbox("Average of (optional)", ["(none)"] + list(df.select_dtypes("number").columns))
    summary = None
    if group != "(none)":
        g = df.groupby(group, dropna=False)
        summary = g.size().rename("rows").to_frame()
        if num != "(none)":
            summary[f"avg {num}"] = g[num].mean().round(2)
        summary = summary.sort_values("rows", ascending=False)
        st.bar_chart(summary["rows"].head(25), horizontal=True)
        st.dataframe(summary, width="stretch")

    if gj is not None:
        metric = st.selectbox("Color the map by", list(df.select_dtypes("number").columns))
        polygon_map(gj, df, metric, [c for c in df.columns[:3]])
    elif (ll := lat_lon_cols(df)) is not None:
        pts = df[[ll[0], ll[1]]].dropna().rename(columns={ll[0]: "lat", ll[1]: "lon"})
        st.map(pts.sample(min(len(pts), 20000), random_state=1), size=20)

    st.subheader("Ask the AI about this table")
    q = st.text_input("Question", "What stands out here, and who in San Antonio would care?")
    if st.button("Ask"):
        table = summary if summary is not None else df.describe(include="all").T
        sent = table.head(40).to_csv()
        st.caption("The AI only sees these numbers:")
        st.code(sent)
        with st.spinner("Thinking..."):
            st.write(ask_ai([
                {"role": "system", "content": "You explain city data to the public in plain English. "
                 "Use only the numbers given. If the numbers can't answer the question, say so."},
                {"role": "user", "content": f"File: {name}\n\n{sent}\n\nQuestion: {q}"}]))

elif page == "311 by neighborhood":
    st.title("311 requests next to the City's equity map")
    req_path = DATA / "311_requests.csv"
    t, _ = load(str(DATA / "tracts_311_and_equity.csv")) if (DATA / "tracts_311_and_equity.csv").exists() else (None, None)
    _, gj = load(str(DATA / "equity_tracts.geojson"))
    if t is None or not req_path.exists():
        st.warning("311 data not downloaded yet. Run tools/fetch_data.py.")
        st.stop()
    t = t.set_index(t["GEOID"].astype(str)).reindex([str(f["properties"]["GEOID"]) for f in gj["features"]]).reset_index(drop=True)
    metrics = {"Late 311 requests (%)": "late_pct", "311 requests": "requests",
               "Median days to close": "median_days_to_close", "Equity score (City, 2-10)": "OVERALL_EQUITY_SCORE",
               "Heat score": "uhi_score", "Median household income": "MED_HSHLD_INCOME"}
    m = st.selectbox("Color by", list(metrics))
    polygon_map(gj, t, metrics[m], ["NAME", "tract_description", "OVERALL_EQUITY_SCORE", "requests", "late_pct"])

    r = pd.read_csv(req_path, usecols=["Category", "TypeName", "LateYesNo", "CouncilDistrict", "DaysToClose", "SourceID"],
                    low_memory=False)
    r["late"] = r["LateYesNo"].str.upper().eq("YES")
    by_d = r.groupby("CouncilDistrict").agg(requests=("late", "size"), late_pct=("late", "mean"),
                                            median_days=("DaysToClose", "median"))
    by_d["late_pct"] = (100 * by_d["late_pct"]).round(1)
    by_d["median_days"] = by_d["median_days"].round(1)
    st.subheader("By council district")
    st.dataframe(by_d, width="stretch")

    d = st.selectbox("District for a brief", sorted(r["CouncilDistrict"].dropna().unique()))
    rd = r[r["CouncilDistrict"] == d]
    types = rd.groupby("TypeName").agg(requests=("late", "size"), late_pct=("late", "mean"))
    types["late_pct"] = (100 * types["late_pct"]).round(1)
    types = types.sort_values("requests", ascending=False).head(10)
    st.dataframe(types, width="stretch")
    lang = st.radio("Language", ["English", "Spanish"], horizontal=True)
    if st.button("Write a one-page brief for this council office"):
        facts = (f"District {d}: {len(rd):,} requests in the last year, {100 * rd['late'].mean():.1f}% closed late "
                 f"(city average {100 * r['late'].mean():.1f}%).\nTop request types:\n{types.to_csv()}")
        st.code(facts)
        with st.spinner("Writing..."):
            st.write(ask_ai([
                {"role": "system", "content": f"Write in {lang}, plain words, for a city council office. "
                 "Use only the numbers given. Under 200 words. End with two concrete next steps."},
                {"role": "user", "content": facts}]))

else:
    st.title("AI helper (runs on this box, no account needed)")
    st.caption("Ask for code, chart ideas, slide wording, or what a column means.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask anything"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Thinking..."):
            answer = ask_ai([{"role": "system", "content": "You help a student team at a one-day data contest. "
                              "Short answers. Python with pandas and streamlit when code is needed."}]
                            + st.session_state.chat, think="medium")
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
