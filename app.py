import streamlit as st
import requests
import re
import nltk
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
from pyvis.network import Network
import os, tempfile

# Download NLTK resources if missing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="🛰️ CTO 2030", layout="wide")

def fetch_hackernews(limit):
    try:
        ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:limit]
        return [requests.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json").json() for i in ids]
    except:
        return []

def fetch_wikipedia(query, limit):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={limit}"
        res = requests.get(url).json()
        return res.get('query', {}).get('search', [])
    except:
        return []

def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(re.sub(r'<.*?>', '', text))
    return r.get_ranked_phrases()

def build_graph_interactive(docs):
    net = Network(height="600px", width="100%", bgcolor="#f9f9f9", font_color="#333")
    net.set_options("{""physics"": {""stabilization"": true}}");
    added_nodes = set()
    for doc in docs:
        keys = extract_keywords(doc.get('title', '') + '. ' + doc.get('content', ''))
        for k in keys:
            if k not in added_nodes:
                net.add_node(k, label=k, color="#90CAF9")
                added_nodes.add(k)
        for i in range(len(keys)-1):
            net.add_edge(keys[i], keys[i+1], color="#B0BEC5")
    return net

def sentiment_summary(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

FMEA_DATA = [
    ["Data Fetch Failure", "No updates from APIs", "API downtime", 8, 5, 4, 160],
    ["Keyword Extraction Error", "No insights generated", "Text parsing issues", 7, 3, 5, 105],
    ["Graph Render Issue", "No relationship visualization", "Rendering error", 6, 2, 4, 48]
]

st.title("🛰️ CTO 2030: AI News & Knowledge Graph Explorer")
st.caption("Real-time tech intelligence with AI insights for the next decade.")

source_choice = st.multiselect("Select Data Sources", ["HackerNews", "Wikipedia"], default=["HackerNews"])
items_per_source = st.slider("Items per source", 1, 20, 3)

all_docs = []
if "HackerNews" in source_choice:
    hn = fetch_hackernews(items_per_source)
    for h in hn:
        all_docs.append({"title": h.get("title", ""), "content": h.get("text", "")})

if "Wikipedia" in source_choice:
    wiki = fetch_wikipedia("Artificial Intelligence", items_per_source)
    for w in wiki:
        all_docs.append({"title": w.get("title", ""), "content": w.get("snippet", "")})

if all_docs:
    st.subheader("📊 Knowledge Graph")
    net = build_graph_interactive(all_docs)
    fd, path = tempfile.mkstemp(suffix=".html")
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    os.close(fd)
    st.components.v1.html(html, height=650)

    combined_text = " ".join([doc['title'] + " " + doc['content'] for doc in all_docs])
    st.subheader("📌 Summary")
    st.success("**Top Keywords:** " + ", ".join(extract_keywords(combined_text)[:10]))

    sentiment = sentiment_summary(combined_text)
    st.subheader("💡 Sentiment Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{sentiment['pos']:.2f}")
    col2.metric("Neutral", f"{sentiment['neu']:.2f}")
    col3.metric("Negative", f"{sentiment['neg']:.2f}")

    st.subheader("🤖 CTO Kumar - Strategic AI Agent")
    st.info("Focus on AI governance, scalable cloud-native systems, quantum-safe security, and ethical data usage.")

    st.subheader("📋 FMEA - Failure Mode and Effects Analysis")
    st.table(FMEA_DATA)
else:
    st.warning("No data to display. Please select at least one source.")
