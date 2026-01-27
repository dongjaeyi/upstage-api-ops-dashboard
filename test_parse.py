import os, time
import requests
from dotenv import load_dotenv
from src.logger import init_db, log_call

load_dotenv()
init_db()

API_KEY = os.getenv("UPSTAGE_API_KEY")
URL = "https://api.upstage.ai/v1/document-digitization"
FILENAME = "data/samples/sample.pdf"

headers = {"Authorization": f"Bearer {API_KEY}"}

with open(FILENAME, "rb") as f:
    files = {"document": f}
    data = {
        "ocr": "force",
        "model": "document-parse",
        "base64_encoding": "['table']",
    }

    t0 = time.time()
    r = requests.post(URL, headers=headers, files=files, data=data)
    dt = int((time.time() - t0) * 1000)

resp_bytes = len(r.content) if r.content else 0

error_code = ""
error_message = ""
if r.status_code >= 400:
    try:
        j = r.json()
        error_code = j.get("error", {}).get("code", "")
        error_message = j.get("error", {}).get("message", "")
    except Exception:
        error_message = r.text[:200]

log_call({
    "endpoint": URL,
    "model": data.get("model"),
    "filename": FILENAME,
    "status_code": r.status_code,
    "latency_ms": dt,
    "response_bytes": resp_bytes,
    "error_code": error_code,
    "error_message": error_message,
})

print("Status:", r.status_code)
print("Latency(ms):", dt)
print("Content-Type:", r.headers.get("Content-Type"))
print("Body:", r.text[:500])

os.makedirs("reports", exist_ok=True)
with open("reports/parse_response.json", "w", encoding="utf-8") as wf:
    wf.write(r.text)
print("Saved: reports/parse_response.json")
