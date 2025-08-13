import streamlit as st
import requests
import re
import nltk
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
from pyvis.network import Network
import tempfile, os
from collections import Counter

# ---------- Setup ----------
# Quiet, one-time downloads to avoid noisy logs
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="🛰️ CTO 2030", layout="wide")

CLEAN_HTML = re.compile(r"<.*?>")

def clean(text: str) -> str:
    if not text:
        return ""
    # Strip HTML tags and collapse whitespace
    return re.sub(r"\s+", " ", CLEAN_HTML.sub(" ", text)).strip()

# ---------- Data fetchers (cached) ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_hackernews(limit: int):
    try:
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10,
        ).json()[:limit]
        docs = []
        for i in ids:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{i}.json",
                timeout=10,
            ).json() or {}
            docs.append({
                "title": item.get("title", ""),
                "content": clean(item.get("text", "")),
                "source": "HackerNews",
            })
        return docs
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_wikipedia(query: str, limit: int):
    try:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            + requests.utils.quote(query)
            + "&format=json&srlimit="
            + str(limit)
        )
        res = requests.get(url, timeout=10).json()
        docs = []
        for w in res.get("query", {}).get("search", []):
            docs.append({
                "title": w.get("title", ""),
                "content": clean(w.get("snippet", "")),
                "source": "Wikipedia",
            })
        return docs
    except Exception:
        return []

# ---------- NLP helpers ----------

def extract_keywords(text: str, max_k: int = 5):
    r = Rake()  # rake-nltk uses NLTK stopwords internally
    r.extract_keywords_from_text(clean(text))
    phrases = [p.strip() for p in r.get_ranked_phrases() if p.strip()]
    # keep concise phrases
    phrases = [p for p in phrases if 2 <= len(p) <= 40][:max_k]
    return phrases

# ---------- Graph builder (PyVis) ----------

def build_graph(docs):
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222")

    # Valid JSON string for PyVis options (previous error came from malformed JSON)
    net.set_options(
        '{"nodes":{"shape":"dot","scaling":{"min":6,"max":28}},'
        '"edges":{"smooth":{"type":"dynamic"},"color":{"color":"#b3c1d1"}},'
        '"physics":{"solver":"barnesHut","stabilization":{"enabled":true,"iterations":250}},'
        '"interaction":{"hover":true,"tooltipDelay":120}}'
    )

    # Aggregate keyword frequencies and co-occurrences
    freq = Counter()
    co = Counter()

    for d in docs:
        keys = extract_keywords(d.get("title", "") + ". " + d.get("content", ""), max_k=5)
        # frequency
        freq.update(keys)
        # co-occurrence within the same doc (unordered pairs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sorted((keys[i], keys[j]))
                co[(a, b)] += 1

    # Add nodes sized by frequency
    for k, c in freq.items():
        size = 8 + min(22, c * 3)
        net.add_node(k, label=k, value=c, title=f"{k} • freq {c}", color="#60a5fa")

    # Add edges weighted by co-occurrence
    for (a, b), w in co.items():
        net.add_edge(a, b, value=w, title=f"co-occur: {w}")

    return net

# ---------- UI ----------
st.title("🛰️ CTO 2030: AI News & Knowledge Graph Explorer")
st.caption("Real-time tech intelligence with AI insights for the next decade.")

colA, colB = st.columns([3, 1])
with colA:
    sources = st.multiselect(
        "Select Data Sources", ["HackerNews", "Wikipedia"], default=["HackerNews", "Wikipedia"]
    )
with colB:
    per_source = st.slider("Items per source", 1, 20, 3)

query = st.text_input("Wikipedia query", value="Artificial Intelligence")

# Collect docs
docs = []
if "HackerNews" in sources:
    docs += fetch_hackernews(per_source)
if "Wikipedia" in sources:
    docs += fetch_wikipedia(query, per_source)

if docs:
    st.subheader("📊 Knowledge Graph")
    net = build_graph(docs)
    fd, path = tempfile.mkstemp(suffix=".html")
    try:
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    finally:
        os.close(fd)
    st.components.v1.html(html, height=680)

    # ---- Summary ----
    all_text = " ".join([d["title"] + " " + d["content"] for d in docs])
    top_keywords = extract_keywords(all_text, max_k=10)

    st.subheader("📌 Summary")
    if top_keywords:
        st.success("**Top Keywords:** " + ", ".join(top_keywords))
    else:
        st.info("No keywords extracted from the current selection.")

    # ---- Sentiment ----
    sia = SentimentIntensityAnalyzer()
    s = sia.polarity_scores(all_text)
    st.subheader("💡 Sentiment Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", f"{s['pos']:.2f}")
    c2.metric("Neutral", f"{s['neu']:.2f}")
    c3.metric("Negative", f"{s['neg']:.2f}")

    # ---- Agent: CTO Kumar ----
    st.subheader("🤖 CTO Kumar – Strategic AI Agent")
    insights = []
    if any("governance" in k or "policy" in k for k in top_keywords):
        insights.append("Establish an AI governance board and model risk framework.")
    if any("edge" in k or "cloud" in k for k in top_keywords):
        insights.append("Adopt edge+cloud patterns with zero-downtime blue/green deploys.")
    if any("privacy" in k or "federated" in k for k in top_keywords):
        insights.append("Invest in privacy-preserving analytics and federated learning pipelines.")
    if not insights:
        insights = [
            "Prioritize scalable cloud-native services with event-driven integration.",
            "Introduce AI-assisted developer tooling in CI/CD for velocity and safety.",
            "Build an enterprise knowledge graph as a semantic layer across data domains.",
        ]
    st.info("\n".join(f"• {i}" for i in insights))

    # ---- FMEA ----
    st.subheader("📋 FMEA – Failure Mode & Effects Analysis")
    fmea_rows = [
        {
            "Failure Mode": "Data Fetch Failure",
            "Effect": "No updates from APIs",
            "Cause": "API downtime / rate limits",
            "Severity": 8,
            "Occurrence": 5,
            "Detection": 4,
            "RPN": 8 * 5 * 4,
        },
        {
            "Failure Mode": "Keyword Extraction Error",
            "Effect": "No insights generated",
            "Cause": "Unexpected HTML / text noise",
            "Severity": 7,
            "Occurrence": 3,
            "Detection": 5,
            "RPN": 7 * 3 * 5,
        },
        {
            "Failure Mode": "Graph Render Issue",
            "Effect": "No relationship visualization",
            "Cause": "PyVis rendering failure",
            "Severity": 6,
            "Occurrence": 2,
            "Detection": 4,
            "RPN": 6 * 2 * 4,
        },
    ]
    st.table(fmea_rows)
else:
    st.warning("No data to display. Adjust sources or increase items per source.")
