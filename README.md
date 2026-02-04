Upstage Resume RAG Summary Builder

Flask-based RAG Pipeline & Ops-Aware Dashboard

This project is a Resume RAG (Retrieval-Augmented Generation) Summary Builder built on top of Upstage APIs, with a strong emphasis on API behavior, system design, and operational awareness rather than just model output quality.

The goal is to demonstrate how a Product Manager can understand and explain:

AI API call flows

Authentication and rate limits

Error handling and observability

Cost and usage implications

Design trade-offs in RAG systems

The entire system is implemented using Flask + Python modules + HTML/CSS, without relying on external LLM frameworks such as LangChain.

What This Project Demonstrates

End-to-end RAG pipeline design using Upstage APIs

Practical use of Document Parse, Embeddings, and LLM completion

Local vector search using FAISS

API-level logging and observability using SQLite

A lightweight Flask web dashboard for monitoring pipeline state and API usage

This mirrors the kind of internal tooling used to:

Debug AI pipelines

Monitor API usage and failures

Reason about rate limits and cost

Explain system behavior to non-ML stakeholders

System Architecture Overview
Core Components

Metadata Store: SQLite (ops.db)

Vector DB: FAISS (local)

Web Server: Flask

Execution Model: CLI pipeline executed via subprocess

Upstage APIs Used

Document Parse

OCR + structured JSON extraction from resumes

Embeddings

Model: embedding-query

Used for resume chunk indexing and JD retrieval

Solar Chat Completions

Used for summary rewriting with guardrails

RAG Pipeline Design
Pipeline Stages

ingest

PDF resume → Document Parse

Structured JSON extraction

Resume metadata and chunks stored in SQLite

index

Resume chunks embedded

FAISS index created locally

retrieve

Job Description embedded

Top-k relevant resume chunks retrieved from FAISS

rewrite_summary

Solar Chat Completion

Guardrail enforced: generation restricted to retrieved chunks only

Output:

Summary text

Citations (chunk_id)

List of used resumes

Execution Modes

pipeline all

Full pipeline execution

Supports:

--skip_ingest

--skip_index (fast iteration when JD changes)

pipeline summary

Runs summary generation only

All pipelines are executed from the repository root using:

python -m src.<pipeline>

Web Dashboard (Flask)

The Flask web application provides operational visibility into the RAG system.

Design Choice

Flask does not import pipeline modules directly

Pipelines are executed via subprocess

This keeps:

CLI and Web concerns cleanly separated

Execution behavior consistent between CLI and UI

Implemented Dashboard Sections
1. Pipeline & AI Architecture Overview

Displays pipeline stages with status:

RUN / SKIPPED / REUSE

Reflects selected execution mode

2. API Usage & Rate Awareness Panel

Key metrics:

Total API calls

Total execution time

Failure count

Hotspot APIs

Error-focused KPIs:

HTTP 429 (rate limit)

5xx errors

Timeouts

API breakdown table

Recent API events (last 10 calls)

Scope toggle:

Last 48 hours

This run (latest execution)

API Logging & Observability
Database: ops.db

The api_calls table captures API-level behavior with fields including:

run_id

stage

api_name

endpoint

model

status_code

latency_ms

ok

error_type

error_message

timestamp

Logging Design Notes

If run_id is missing in the payload, a default process-level RUN_ID is assigned

Embedding calls are logged at the API boundary, not just at pipeline level

Current this run grouping is process-based

Sufficient for demos and PM explanation

Can be extended by passing RUN_ID via subprocess environment variables

Project Structure
.
├── src/

│   ├── pipeline/           # ingest / index / retrieve / summary

│   ├── web_app.py          # Flask web server

│   ├── upstage_client.py   # API wrappers

│   └── logger.py           # SQLite logging utilities

├── data/
│   └── samples/            # Sample resumes (not included)

├── outputs/                # Generated summaries (gitignored)

├── reports/                # Raw API responses (gitignored)

├── app.py                  # Flask entry point

├── requirements.txt

├── .env.example

└── README.md


Setup (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt


Create a .env file in the project root:

UPSTAGE_API_KEY=your_upstage_api_key


.env is intentionally excluded from version control.

How to Run
CLI Pipeline
python -m src.pipeline all


Fast iteration:

python -m src.pipeline all --skip_ingest --skip_index

Web Dashboard
python app.py


Open the displayed local URL to access the dashboard.

Design Philosophy

Prefer explicit system design over hidden abstractions

Treat AI APIs as production systems, not black boxes

Optimize for explainability and observability

Keep the project understandable for PMs and non-ML stakeholders

Possible Next Extensions

Cost and billing estimation per run

Config trade-off comparison (chunk size, top-k, model choice)

Feedback loop from output quality → retrieval tuning

Run-level lineage tracking across subprocess calls

License

This project is provided as-is for demonstration and educational purposes.
