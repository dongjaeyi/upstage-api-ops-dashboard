import os, time, base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("UPSTAGE_API_KEY", "")
DP_ENDPOINT = os.getenv("DP_ENDPOINT")
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT")
IE_ASYNC_ENDPOINT = os.getenv("IE_ASYNC_ENDPOINT")

def _headers_json():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def _headers_auth_only():
    return {"Authorization": f"Bearer {API_KEY}"}

def call_document_parse(file_path: str, model: str = "document-parse-nightly", mode: str = "auto"):
    """
    Uses multipart/form-data as shown in Upstage examples. :contentReference[oaicite:8]{index=8}
    """
    t0 = time.time()
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {"model": model, "mode": mode}
        r = requests.post(DP_ENDPOINT, headers=_headers_auth_only(), files=files, data=data, timeout=120)
    dt = time.time() - t0
    return r.status_code, dt, r.json() if "application/json" in r.headers.get("Content-Type","") else r.text

def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # png/jpg만 우선 지원(면접용으로 충분)
    mime = "image/png" if image_path.lower().endswith("png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def call_information_extract_via_chat(image_path: str, json_schema: dict):
    """
    Information Extract 소개 글에서 OpenAI-style messages + json_schema 형태가 나옵니다. :contentReference[oaicite:9]{index=9}
    """
    payload = {
        "model": "information-extract",
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}]
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "doc_schema", "schema": json_schema}
        }
    }
    t0 = time.time()
    r = requests.post(CHAT_ENDPOINT, headers=_headers_json(), json=payload, timeout=120)
    dt = time.time() - t0
    return r.status_code, dt, r.json()

def call_information_extract_async(messages: list, response_format: dict):
    """
    Async endpoint는 공식 블로그에 예시가 있습니다. :contentReference[oaicite:10]{index=10}
    """
    payload = {"model": "information-extract", "messages": messages, "response_format": response_format}
    r = requests.post(IE_ASYNC_ENDPOINT, headers=_headers_json(), json=payload, timeout=120)
    return r.status_code, r.json()
