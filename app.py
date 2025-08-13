import nltk_setup  # Ensures punkt data is available before running the app
import streamlit as st
from rake_nltk import Rake
import networkx as nx
from pyvis.network import Network
import requests

st.set_page_config(page_title="🛰️ CTO 2030: AI News & Knowledge Graph Explorer", layout="wide")

st.title("🛰️ CTO 2030: AI News & Knowledge Graph Explorer")
st.write("Real-time tech intelligence with AI insights for the next decade.")

source = st.multiselect("Select Data Sources", ["HackerNews", "Wikipedia"])
items_per_source = st.slider("Items per source", 1, 20, 5)

query = st.text_input("Wikipedia query", "Artificial Intelligence")

if st.button("📊 Knowledge Graph"):
    docs = []
    if "Wikipedia" in source:
        res = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}").json()
        docs.append({"title": res.get("title", ""), "content": res.get("extract", "")})

    if docs:
        r = Rake()
        G = nx.Graph()
        for d in docs:
            r.extract_keywords_from_text(d.get("title", "") + ". " + d.get("content", ""))
            for kw in r.get_ranked_phrases():
                G.add_node(kw)
                G.add_edge(d.get("title", ""), kw)

        net = Network(notebook=False, height="500px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        net.show("graph.html")
        st.components.v1.html(open("graph.html", "r", encoding="utf-8").read(), height=520)
