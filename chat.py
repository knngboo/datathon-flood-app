"""Flood photo chat: pick an aerial photo, see it (and its before/after pair), and ask the AI about it.

Run on the Dell:  streamlit run chat.py --server.address 0.0.0.0 --server.port 8501
"""
import base64
import io
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "Texas_Flood_Student_Data"))
AI_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")
AI_MODEL = os.environ.get("VLM_MODEL", "/home/dell/models/qwen3-VL-8b-instruct")
SYSTEM = ("You help emergency responders and residents understand aerial flood photos. "
          "Answer in plain words, briefly. Say what you can see and what you cannot tell from the photo. "
          "A normal river, pond or pool is not a flood. If two photos are given, the first is before and the second is after.")

st.set_page_config(page_title="Flood photo chat", layout="wide")


@st.cache_data
def photos():
    df = pd.read_csv(DATA / "image_metadata.csv")
    df = df[df["image_path"].map(lambda p: (DATA / p).exists())]
    df["label"] = (df["event_name"] + " | " + df["locality"].fillna("?") + " | "
                   + df["acquisition_date"].astype(str) + " | " + df["image_role"] + " | " + df["image_id"])
    return df


def data_url(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(messages):
    body = {"model": AI_MODEL, "messages": messages, "max_tokens": 400, "temperature": 0.2}
    req = urllib.request.Request(AI_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer NONE"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"The AI didn't answer: {e}"


df = photos()
st.sidebar.title("Pick a photo")
event = st.sidebar.selectbox("Flood", sorted(df["event_name"].unique()))
role = st.sidebar.radio("Show", ["response", "before"], format_func={"response": "After the flood", "before": "Before"}.get)
choices = df[(df["event_name"] == event) & (df["image_role"] == role)].sort_values("image_id")
pick = st.sidebar.selectbox("Photo", choices["label"])
row = choices[choices["label"] == pick].iloc[0]
pair = df[df["image_id"] == row["paired_image_id"]]
compare = st.sidebar.checkbox("Compare with its pair", value=False, disabled=pair.empty)

shown = [row]
if compare and not pair.empty:
    shown = sorted([row, pair.iloc[0]], key=lambda r: r["acquisition_date"])

key = "|".join(r["image_id"] for r in shown)
if st.session_state.get("key") != key:
    st.session_state.key, st.session_state.chat = key, []

st.title("Flood photo chat")
cols = st.columns(len(shown))
for c, r in zip(cols, shown):
    c.image(str(DATA / r["image_path"]), width="stretch",
            caption=f"{r['image_role'].upper()} | {r['acquisition_date']} | {r['locality']} | {r['source_agency']}")
st.caption(f"Location: {row['latitude']:.5f}, {row['longitude']:.5f}")

for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["text"])

starters = ["Is floodwater covering normally dry land? Give one visible clue.",
            "Are any roads or houses under water?",
            "What changed between the two photos?" if len(shown) == 2 else "Describe this area in two sentences."]
cols = st.columns(len(starters))
clicked = next((s for c, s in zip(cols, starters) if c.button(s)), None)
question = st.chat_input("Ask about this photo") or clicked

if question:
    st.session_state.chat.append({"role": "user", "text": question})
    st.chat_message("user").write(question)
    # The photos go with the first question; the whole chat is resent every time so the AI remembers it.
    msgs = [{"role": "system", "content": SYSTEM}]
    for i, m in enumerate(st.session_state.chat):
        if i == 0:
            content = [{"type": "image_url", "image_url": {"url": data_url(DATA / r["image_path"])}} for r in shown]
            content.append({"type": "text", "text": m["text"]})
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": m["role"], "content": m["text"]})
    with st.spinner("Looking at the photo..."):
        answer = ask(msgs)
    st.session_state.chat.append({"role": "assistant", "text": answer})
    st.chat_message("assistant").write(answer)
