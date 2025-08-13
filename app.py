import streamlit as st
import requests
import re
import nltk
from rake_nltk import Rake
import networkx as nx
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are downloaded
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'vader_lexicon']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

st.set_page_config(page_title="🛰️ CTO 2030", layout="wide")

# Fetch data functions
def fetch_hackernews(limit):
    try:
        ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:limit]
        return [requests.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json").json() for i in ids]
    except Exception:
        return []

def fetch_wikipedia(query, limit):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={limit}"
        res = requests.get(url).json()
        return res.get('query', {}).get('search', [])
    except Exception:
        return []

# Keyword extraction
def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(re.sub(r'<.*?>', '', text))  # remove HTML tags
    return r.get_ranked_phrases()

# Build knowledge graph
def build_graph(docs):
    G = nx.Graph()
    for doc in docs:
        keys = extract_keywords(doc.get('title', '') + '. ' + doc.get('content', ''))
        for k in keys:
            G.add_node(k)
        for i in range(len(keys)-1):
            G.add_edge(keys[i], keys[i+1])
    return G

# Sentiment analysis
def sentiment_summary(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# FMEA data
FMEA_HEADERS = ["Failure Mode", "Effect", "Cause", "Severity", "Occurrence", "Detection", "RPN"]
FMEA_DATA = [
    ["Data Fetch Failure", "No updates from APIs", "API downtime", 8, 5, 4, 160],
    ["Keyword Extraction Error", "No insights generated", "Text parsing issues", 7, 3, 5, 105],
    ["Graph Render Issue", "No relationship visualization", "Rendering error", 6, 2, 4, 48]
]

# UI Layout
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
    G = build_graph(all_docs)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=8)
    st.pyplot(plt)

    combined_text = " ".join([doc['title'] + " " + doc['content'] for doc in all_docs])
    st.subheader("Summary")
    st.write("**Top Keywords:**", ", ".join(extract_keywords(combined_text)[:10]))
    st.write("**Sentiment:**", sentiment_summary(combined_text))

    st.subheader("🤖 CTO Kumar - Strategic AI Agent")
    st.info("CTO Kumar recommends focusing on AI governance, scalable cloud-native systems, quantum-safe security, and ethical data usage for future-proof architectures.")

    st.subheader("FMEA - Failure Mode and Effects Analysis")
    st.table(FMEA_DATA)
else:
    st.warning("No data to display. Please select at least one source.")
