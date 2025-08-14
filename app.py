import streamlit as st
from pyvis.network import Network
import wikipedia
import feedparser
import urllib.parse
from transformers import pipeline

# ---------------------------
# APP CONFIG
# ---------------------------
st.set_page_config(page_title="🛰️ CTO 2030: AI Multi-Source Knowledge Graph", layout="wide")
st.title("🛰️ CTO 2030: AI Multi-Source Knowledge Graph")
st.caption("Real-time tech intelligence with AI insights for the next decade.")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Select Data Sources")
wiki_enabled = st.sidebar.checkbox("Wikipedia", value=True)
arxiv_enabled = st.sidebar.checkbox("ArXiv AI Research", value=True)
google_enabled = st.sidebar.checkbox("Google News", value=True)
hf_enabled = st.sidebar.checkbox("Hugging Face Models", value=True)

items_per_source = st.sidebar.slider("Items per source", 1, 20, 5)
search_query = st.sidebar.text_input("Search Query", "Artificial Intelligence")

# Encode for safe URL use
encoded_query = urllib.parse.quote_plus(search_query)

# ---------------------------
# KNOWLEDGE GRAPH
# ---------------------------
net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
net.add_node(search_query, label=search_query, color="#ff0000")

all_nodes = set()
all_nodes.add(search_query)

# ---------------------------
# WIKIPEDIA
# ---------------------------
if wiki_enabled and search_query.strip():
    try:
        st.subheader("📚 Wikipedia")
        wiki_results = wikipedia.search(search_query, results=items_per_source)
        st.write(wiki_results)
        for title in wiki_results:
            net.add_node(title, label=title, color="#00ff00")
            net.add_edge(search_query, title)
            all_nodes.add(title)
    except Exception as e:
        st.error(f"Wikipedia error: {e}")

# ---------------------------
# ARXIV
# ---------------------------
if arxiv_enabled:
    try:
        st.subheader("📄 ArXiv Research")
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={items_per_source}"
        feed = feedparser.parse(arxiv_url)
        arxiv_titles = [entry.title for entry in feed.entries]
        st.write(arxiv_titles)
        for title in arxiv_titles:
            net.add_node(title, label=title, color="#ffaa00")
            net.add_edge(search_query, title)
            all_nodes.add(title)
    except Exception as e:
        st.error(f"ArXiv error: {e}")

# ---------------------------
# GOOGLE NEWS
# ---------------------------
if google_enabled:
    try:
        st.subheader("📰 Google News")
        google_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(google_url)
        news_titles = [entry.title for entry in feed.entries]
        st.write(news_titles[:items_per_source])
        for title in news_titles[:items_per_source]:
            net.add_node(title, label=title, color="#00aaff")
            net.add_edge(search_query, title)
            all_nodes.add(title)
    except Exception as e:
        st.error(f"Google News error: {e}")

# ---------------------------
# HUGGING FACE MODELS
# ---------------------------
if hf_enabled:
    try:
        st.subheader("🤗 Hugging Face Models")
        hf_models = [
            "MennaAllahreda25/ArtificialIntelligenceModels",
            "ArtificialIntellect/cat-breed-classifier",
            "twiarshwest/artificial-intelligence-interior-assistant",
            "MSDDSDSDSAAAAAA/Consumer-behavior-analysis-with-artificial-intelligence",
            "Applied-Artificial-Intelligence-Eurecat/IMPETUS-Climate-bge-small"
        ]
        st.write(hf_models[:items_per_source])
        for model in hf_models[:items_per_source]:
            net.add_node(model, label=model, color="#ff00ff")
            net.add_edge(search_query, model)
            all_nodes.add(model)
    except Exception as e:
        st.error(f"Hugging Face Models error: {e}")

# ---------------------------
# DISPLAY KNOWLEDGE GRAPH
# ---------------------------
net.save_graph("graph.html")
with open("graph.html", "r", encoding="utf-8") as f:
    html_code = f.read()
st.components.v1.html(html_code, height=550, scrolling=True)

# ---------------------------
# AGENTIC AI SUMMARIES
# ---------------------------
st.subheader("🧠 AI Agent Summaries for Knowledge Graph Nodes")

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    for node in all_nodes:
        try:
            # Fetch small Wikipedia snippet if available
            try:
                page = wikipedia.page(node, auto_suggest=False)
                text = page.content[:500]  # small chunk to save resources
            except:
                text = node  # fallback to node text

            summary = summarizer(text, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
            st.write(f"**{node}:** {summary}")
        except Exception as e:
            st.write(f"**{node}:** (No summary available)")

except Exception as e:
    st.error(f"AI summarization error: {e}")
