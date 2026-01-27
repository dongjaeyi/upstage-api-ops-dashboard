import os, time, json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("UPSTAGE_API_KEY")  # 너 .env에 저장한 키 이름에 맞춰
URL = "https://api.upstage.ai/v1/document-digitization"

FILENAME = "data/samples/sample.pdf"  # 같은 폴더에 샘플 이미지 하나 두고 이름 맞추기

headers = {"Authorization": f"Bearer {API_KEY}"}

with open(FILENAME, "rb") as f:
    files = {"document": f}
    data = {
        "ocr": "force",
        "model": "document-parse",
        # 일단 이 줄은 있어도 되고 없어도 됨. 예시는 table base64 인코딩.
        "base64_encoding": "['table']",
    }

    t0 = time.time()
    r = requests.post(URL, headers=headers, files=files, data=data)
    dt = int((time.time() - t0) * 1000)

print("Status:", r.status_code)
print("Latency(ms):", dt)
print("Content-Type:", r.headers.get("Content-Type"))
print("Body:", r.text[:500])  # 너무 길면 앞부분만

os.makedirs("reports", exist_ok=True)
with open("reports/parse_response.json", "w", encoding="utf-8") as wf:
    wf.write(r.text)
print("Saved: reports/parse_response.json")

from src.logger import init_db, log_call

init_db()

# ... requests 호출 후
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
