# 📡 Live Sentiment Analyzer — Cloud Demo Version

This is a **lightweight, self-contained** version of the project, built specifically
to deploy on **Streamlit Community Cloud** for a live resume demo link.

It simulates real-time streaming data directly inside the app (no Kafka or
external API needed), so it deploys instantly with zero configuration.

> 📌 The **full production pipeline** (Kafka + HuggingFace RoBERTa + NewsAPI +
> Twitter + SQLite) lives in the main project folder — this cloud version exists
> purely so anyone can click a link and see a working live demo without setup.

---

## Deploy in 3 minutes

1. Push this folder's contents to a GitHub repo
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click **New app** → select your repo → set main file path to `dashboard/app.py`
4. Click **Deploy**

You'll get a permanent link like:
```
https://your-app-name.streamlit.app
```

Put that link on your resume! 🎉

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
