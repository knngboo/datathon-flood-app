import base64, io, json, random, csv, urllib.request, time
from pathlib import Path
from PIL import Image
D = Path.home() / "Texas_Flood_Student_Data"
rows = [r for r in csv.DictReader(open(D / "image_metadata.csv")) if r["event_name"] == "Hurricane Harvey" and r["paired_image_id"]]
after = {r["image_id"]: r for r in rows if r["image_role"] == "response"}
pairs = [(after[i], next(x for x in rows if x["image_id"] == after[i]["paired_image_id"])) for i in after if any(x["image_id"] == after[i]["paired_image_id"] for x in rows)]
random.Random(42).shuffle(pairs); pairs = pairs[:15]
def url(r):
    im = Image.open(D / r["image_path"]).convert("RGB"); im.thumbnail((1024, 1024)); b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
Q = "This is an aerial photo. Is floodwater visible covering streets, yards or around homes? Answer with one word first: Yes, No, or Unclear."
def ask(r):
    body = {"model": "/home/dell/models/qwen3-VL-8b-instruct", "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(r)}}, {"type": "text", "text": Q}]}]}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip().lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else "unclear"
res = []
for a, b in pairs:
    res.append({"pair": a["image_id"], "place": a["locality"], "after_2017": ask(a), "before_2016": ask(b)})
    Path(Path.home() / "datathon-flood-app/e2e/photo-eval.json").write_text(json.dumps(res, indent=1))
n = len(res)
print(f"{n} Harvey photo pairs")
print("2016 photos (before Harvey): AI said NO flood on", sum(r["before_2016"] == "no" for r in res), "of", n, "| false alarms:", sum(r["before_2016"] == "yes" for r in res))
print("2017 photos (Harvey response): AI saw flooding on", sum(r["after_2017"] == "yes" for r in res), "of", n)
