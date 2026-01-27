import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Upstage API Ops Dashboard", layout="wide")
st.title("Upstage API Ops Dashboard (Document Parse)")

DB_PATH = "ops.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM api_calls ORDER BY id DESC", conn)
conn.close()

if df.empty:
    st.info("No logs yet. Run test_parse.py first to generate calls.")
    st.stop()

success_rate = (df["status_code"] < 400).mean() * 100
p95_latency = int(df["latency_ms"].quantile(0.95)) if "latency_ms" in df else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Calls", len(df))
c2.metric("Success Rate", f"{success_rate:.1f}%")
c3.metric("P95 Latency (ms)", p95_latency)

st.subheader("Recent Calls (latest 50)")
st.dataframe(df.head(50), width=True)

st.subheader("Status Code Distribution")
status_counts = df["status_code"].value_counts().sort_index()
st.bar_chart(status_counts)

st.subheader("Latency (ms) over time (latest 200)")
st.line_chart(df.head(200)[::-1].set_index("id")["latency_ms"])
