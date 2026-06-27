"""
dashboard/app.py — Cloud-deploy-ready version
Self-contained: simulates live streaming data directly in-app using
Streamlit session state (no Kafka, no external DB file needed).
Works perfectly on Streamlit Community Cloud.
"""

import time
import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Live Sentiment Analyzer",
    page_icon="📡",
    layout="wide",
)

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral":  "#f59e0b",
    "negative": "#ef4444",
}

MOCK_DATA = {
    "AI": [
        ("AI is transforming healthcare, education and finance at an incredible pace!", "positive"),
        ("Worried about job losses due to AI automation. Very concerning trend.", "negative"),
        ("AI researchers published a new paper on large language models today.", "neutral"),
        ("This new AI tool is absolutely amazing, saves me hours every day!", "positive"),
        ("AI bias in hiring tools is a serious ethical problem we must address.", "negative"),
        ("AI conference scheduled for next month in San Francisco.", "neutral"),
    ],
    "Python": [
        ("Python 3.13 just dropped and the performance improvements are insane!", "positive"),
        ("Python is too slow for production systems. GIL is still a problem.", "negative"),
        ("Updated the Python tutorial for beginners, covers lists and dicts.", "neutral"),
        ("Love how readable Python code is compared to other languages!", "positive"),
        ("Python packaging is a mess. pip, conda, poetry — which one to use?", "negative"),
    ],
    "DataScience": [
        ("Data science is the best career of the decade, amazing salaries!", "positive"),
        ("Tired of cleaning data all day. 80% of DS is just pandas wrangling.", "negative"),
        ("Published a new notebook on kaggle — customer churn prediction.", "neutral"),
        ("Data science skills are in huge demand across every industry!", "positive"),
        ("Another overhyped data science startup just collapsed. Not surprised.", "negative"),
    ],
}


def new_record(ts: datetime) -> dict:
    keyword = random.choice(list(MOCK_DATA.keys()))
    text, label = random.choice(MOCK_DATA[keyword])
    return {
        "text": text,
        "source": random.choice(["news", "twitter"]),
        "keyword": keyword,
        "label": label,
        "score": round(random.uniform(0.7, 0.99), 4),
        "timestamp": ts,
    }


def init_state():
    if "records" not in st.session_state:
        # Seed with backdated records spread over the last 10 minutes
        now = datetime.utcnow()
        st.session_state.records = [
            new_record(now - timedelta(seconds=random.randint(0, 600)))
            for _ in range(40)
        ]
        st.session_state.last_tick = time.time()


def maybe_add_record():
    """Add a new record roughly every refresh, simulating live streaming."""
    now = time.time()
    if now - st.session_state.last_tick > 2:
        st.session_state.records.append(new_record(datetime.utcnow()))
        st.session_state.last_tick = now
        # Cap history so it doesn't grow forever
        if len(st.session_state.records) > 500:
            st.session_state.records = st.session_state.records[-500:]


def get_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.records)


# ── visual components ───────────────────────────────────────────────────────

def kpi_cards(df: pd.DataFrame):
    total = len(df)
    pos_pct = round(100 * (df["label"] == "positive").sum() / max(total, 1))
    neg_pct = round(100 * (df["label"] == "negative").sum() / max(total, 1))
    avg_conf = round(df["score"].mean() * 100, 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analyzed", total)
    c2.metric("Positive %", f"{pos_pct}%")
    c3.metric("Negative %", f"{neg_pct}%")
    c4.metric("Avg confidence", f"{avg_conf}%")


def sentiment_pie(df: pd.DataFrame):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    fig = px.pie(
        counts, values="count", names="label",
        color="label", color_discrete_map=SENTIMENT_COLORS,
        hole=0.45, title="Overall sentiment distribution",
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


def sentiment_over_time(df: pd.DataFrame):
    if df.empty:
        return
    df2 = df.copy()
    df2["minute"] = pd.to_datetime(df2["timestamp"]).dt.floor("1min")
    timeline = df2.groupby(["minute", "label"]).size().reset_index(name="count")
    fig = px.line(
        timeline, x="minute", y="count", color="label",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment volume over time",
        labels={"minute": "Time", "count": "Messages"},
    )
    fig.update_layout(margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def by_keyword_chart(df: pd.DataFrame):
    if df.empty:
        return
    grouped = df.groupby(["keyword", "label"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="keyword", y="count", color="label",
        color_discrete_map=SENTIMENT_COLORS, barmode="group",
        title="Sentiment by keyword",
    )
    fig.update_layout(margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def word_cloud(df: pd.DataFrame, label: str):
    sub = df[df["label"] == label]
    if sub.empty:
        st.info(f"No {label} messages yet.")
        return
    text = " ".join(sub["text"].tolist())
    wc = WordCloud(
        width=600, height=300, background_color="white",
        colormap="RdYlGn" if label == "positive" else "Reds",
        max_words=80,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def latest_feed(df: pd.DataFrame):
    if df.empty:
        st.info("No data yet.")
        return
    recent = df.sort_values("timestamp", ascending=False).head(10)
    for _, row in recent.iterrows():
        color = SENTIMENT_COLORS.get(row["label"], "#888")
        badge = (
            f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:12px">{row["label"].upper()}</span>'
        )
        st.markdown(
            f'{badge} &nbsp;**[{row["source"]} · {row["keyword"]}]** &nbsp;'
            f'<span style="font-size:13px">{row["text"][:140]}</span>',
            unsafe_allow_html=True,
        )
        st.divider()


# ── layout ───────────────────────────────────────────────────────────────

st.title("📡 Live Sentiment Analyzer")
st.caption(
    "Real-time sentiment tracking pipeline — News + Twitter, "
    "RoBERTa NLP model, Kafka streaming architecture. "
    "(This hosted demo uses simulated live data; see GitHub for the full Kafka + HuggingFace pipeline.)"
)

with st.sidebar:
    st.header("⚙️ Controls")
    keyword_filter = st.selectbox("Filter by keyword", ["All", "AI", "Python", "DataScience"])
    source_filter = st.selectbox("Source", ["All", "news", "twitter"])
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("News/Twitter → Kafka → RoBERTa → DB → Dashboard")
    st.markdown("**Model**")
    st.markdown("`cardiffnlp/twitter-roberta-base-sentiment-latest`")
    st.markdown("---")
    st.markdown("[View full source on GitHub](https://github.com/meharkaur02/live-sentiment-analyzer)")
init_state()
maybe_add_record()

df = get_df()
if keyword_filter != "All":
    df = df[df["keyword"] == keyword_filter]
if source_filter != "All":
    df = df[df["source"] == source_filter]

kpi_cards(df)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    sentiment_pie(df)
with col2:
    sentiment_over_time(df)

st.markdown("---")
by_keyword_chart(df)

st.markdown("---")
st.subheader("☁️ Word clouds")
wc1, wc2 = st.columns(2)
with wc1:
    st.markdown("**Positive messages**")
    word_cloud(df, "positive")
with wc2:
    st.markdown("**Negative messages**")
    word_cloud(df, "negative")

st.markdown("---")
st.subheader("📋 Live feed")
latest_feed(df)

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC — auto-refreshing every few seconds")

time.sleep(3)
st.rerun()
