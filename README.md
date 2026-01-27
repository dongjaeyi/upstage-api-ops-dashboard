# Upstage API Ops Dashboard (Document Parse)

Mini ops dashboard for Upstage Document Digitization (document-parse).
Tracks API latency, status codes, and error patterns in SQLite and visualizes them in Streamlit.

## Setup (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install requests python-dotenv streamlit pandas
