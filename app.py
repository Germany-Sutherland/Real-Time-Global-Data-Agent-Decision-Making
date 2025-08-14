import streamlit as st
from pyvis.network import Network
import wikipedia
import feedparser
import requests

# --- App Title ---
st.set_page_config(page_title="🛰️ CTO 2030: AI Multi-Source Knowledge Graph", layout="wide")
st.title("🛰️ CTO 2030: AI Multi-Source Knowledge Graph")
st.caption("Real-time tech intelligence with AI insights for the next decade.")

# --- Sidebar ---
st.sidebar.header("Select Data Sources")
wiki_enabled = st.sidebar.checkbox("Wikipedia", value=True)
arxiv_enabled = st.sidebar.checkbox("ArXiv AI Research", value=True)
news_enabled = st.sidebar.checkbox("Google News", value=True)
hf_enabled = st.sidebar.checkbox("Hugging Face Models", value=True)

items_per_source = st.sidebar.slider("Items per source", 1, 20, 5)
search_query = st.sidebar.text_input("Search Query", "Artificial Intelligence")

# --- Create PyVis Graph ---
net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
net.add_node(search_query, label=search_query, color="#ff0000")

# --- Wikipedia ---
if wiki_enabled and search_query.strip():
    try:
        st.subheader("📚 Wikipedia")
        search_results = wikipedia.search(search_query, results=items_per_source)
        st.write(search_results)
        for title in search_results:
            net.add_node(title, label=title, color="#00ff00")
            net.add_edge(search_query, title)
    except Exception as e:
        st.error(f"Wikipedia error: {e}")

# --- ArXiv AI Research ---
if arxiv_enabled and search_query.strip():
    try:
        st.subheader("📝 ArXiv Research")
        feed_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={items_per_source}"
        feed = feedparser.parse(feed_url)
        arxiv_titles = [entry.title for entry in feed.entries]
        st.write(arxiv_titles)
        for title in arxiv_titles:
            net.add_node(title, label=title, color="#00ffff")
            net.add_edge(search_query, title)
    except Exception as e:
        st.error(f"ArXiv error: {e}")

# --- Google News RSS ---
if news_enabled and search_query.strip():
    try:
        st.subheader("📰 Google News")
        rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        news_titles = [entry.title for entry in feed.entries[:items_per_source]]
        st.write(news_titles)
        for title in news_titles:
            net.add_node(title, label=title, color="#ffff00")
            net.add_edge(search_query, title)
    except Exception as e:
        st.error(f"Google News error: {e}")

# --- Hugging Face Models ---
if hf_enabled and search_query.strip():
    try:
        st.subheader("🤗 Hugging Face Models")
        url = f"https://huggingface.co/api/models?search={search_query}&limit={items_per_source}"
        resp = requests.get(url)
        if resp.status_code == 200:
            models = [m['modelId'] for m in resp.json()]
            st.write(models)
            for model in models:
                net.add_node(model, label=model, color="#ff00ff")
                net.add_edge(search_query, model)
    except Exception as e:
        st.error(f"Hugging Face error: {e}")

# --- Display Graph ---
try:
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    st.components.v1.html(html_code, height=550, scrolling=True)
except Exception as e:
    st.error(f"Graph error: {e}")

# --- Example for CTO ---
st.markdown("---")
st.markdown("""
**💡 Example for a CTO (2030 Scenario)**  
Let’s say you’re deciding whether to invest in **Agentic AI**:  
- You enter `"Agentic AI"` → The app shows related tech from Wikipedia, latest AI research from ArXiv, real-time news headlines, and relevant Hugging Face models.  
- You quickly spot key components and tools you might need to integrate.  
- You also see emerging related topics you hadn’t considered.  

✅ This shortens your research from **days to minutes**.
""")
