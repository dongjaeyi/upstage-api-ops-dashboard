# Upstage API Ops Dashboard (Document Parse)

A lightweight operations dashboard built on top of **Upstage Document Digitization (Document Parse)** API.

This project demonstrates how to:
- Call the Upstage Document Parse API
- Log API behavior such as latency and status codes
- Store request results locally
- Visualize operational metrics using Streamlit

The goal is to provide a simple but practical example of **API-level observability** for document AI workflows.

---

## Features

### Document Parse API Integration
- Sends PDF or image documents to the Upstage Document Digitization API
- Uses multipart form requests as required by the API
- Captures response metadata for analysis

### Local API Call Logging
Each API request is logged to a local SQLite database, including:
- Timestamp
- Endpoint
- Model name
- File name
- HTTP status code
- Latency (ms)
- Response size
- Error code and message (if any)

### Streamlit Dashboard
A Streamlit-based dashboard provides visibility into:
- Total API calls
- Success rate
- P95 latency
- Status code distribution
- Latency trends over time

---

## Project Structure
├── app.py # Streamlit dashboard
├── test_parse.py # API call and logging script
├── src/
│ └── logger.py # SQLite logging utilities
├── data/
│ └── samples/ # Sample documents (not included)
├── reports/ # Sample API responses (optional)
├── requirements.txt
├── .env.example
└── README.md

---

## Requirements

- Python 3.9+
- Upstage API key

---

## Setup (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

Create a .env file in the project root:

UPSTAGE_API_KEY=your_upstage_api_key


Note: .env is intentionally excluded from version control.

How to Run
1. Generate API call logs
python test_parse.py


This will:

Send a document to the Upstage Document Parse API

Save the raw response to reports/

Log request metadata to ops.db

2. Launch the dashboard
streamlit run app.py


Open the displayed local URL in your browser to view the dashboard.

Notes

This project focuses on API behavior and observability, not model training.

Sample documents should not contain sensitive or personal information.

The SQLite database (ops.db) is generated locally and can be safely deleted at any time.

Possible Extensions

Batch or load testing for rate-limit analysis

Additional metrics such as error frequency by document type

Integration with Information Extraction APIs

Exporting metrics for external monitoring systems

License

This project is provided as-is for demonstration and educational purposes.