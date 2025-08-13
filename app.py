import re
import time
from typing import List, Dict, Tuple

import requests
import pandas as pd
import networkx as nx
from bs4 import BeautifulSoup
from pyvis.network import Network

import streamlit as st
import streamlit.components.v1 as components

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

# NLTK downloads (silent)
for pkg in ["vader_lexicon", "punkt", "stopwords"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

st.set_page_config(
    page_title="CTO 2030: AI News & Knowledge Graph Explorer",
    page_icon="🛰️",
    layout="wide"
)

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\\+?\\d{1,3}[\\s-]?)?(\\(?\\d{3}\\)?[\\s-]?)?\\d{3}[\\s-]?\\d{4}")
BIAS_LEXICON = {
    'loaded': [
        'obviously','clearly','undeniably','disgraceful','shocking','outrageous',
        'fake','propaganda','agenda','witch-hunt','radical','extremist','corrupt','traitor'
    ]
}

def redact_privacy(text: str) -> str:
    return PHONE_RE.sub('[REDACTED_PHONE]', EMAIL_RE.sub('[REDACTED_EMAIL]', text))

def simple_summarize(text: str, max_sentences: int = 3) -> str:
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, s in enumerate(sentences):
        wds = [w.lower() for w in word_tokenize(s) if w.isalpha()]
        score = sum(freq.get(w, 0) for w in wds) / (len(wds) + 1e-6)
        scores.append((i, score))
    keep_idx = sorted([i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]])
    return ' '.join(sentences[i] for i in keep_idx)

def extract_keywords(text: str, max_phrases: int = 8) -> List[str]:
    r = Rake()
    r.extract_keywords_from_text(text)
    return [p for p, _ in r.get_ranked_phrases_with_scores()[:max_phrases] if 2 <= len(p.split()) <= 6]

def analyze_bias(text: str) -> Dict[str, float]:
    sia = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(text)
    lower = text.lower()
    loaded_terms = sum(lower.count(w) for w in BIAS_LEXICON['loaded'])
    words = max(1, len(lower.split()))
    bias = 1000.0 * loaded_terms / words
    sent['bias_loaded_terms_per_1k_words'] = round(bias, 3)
    return sent

def fetch_hn_frontpage(n: int = 10) -> List[Dict]:
    url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={n}"
    r = requests.get(url, timeout=20)
    data = r.json().get('hits', [])
    return [{
        'source': 'HackerNews',
        'title': h.get('title', ''),
        'url': h.get('url', ''),
        'content': h.get('story_text') or h.get('title', ''),
        'timestamp': h.get('created_at')
    } for h in data]

def fetch_wikipedia_recent(n: int = 10) -> List[Dict]:
    rc_url = ("https://en.wikipedia.org/w/api.php?action=query&list=recentchanges" 
              f"&rcprop=title|timestamp&rclimit={n}&format=json")
    r = requests.get(rc_url, timeout=20)
    changes = r.json().get('query', {}).get('recentchanges', [])
    items = []
    for c in changes:
        title = c.get('title', '')
        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        s = requests.get(sum_url, timeout=20)
        if s.status_code == 200:
            js = s.json()
            extract = js.get('extract', '')
            url = js.get('content_urls', {}).get('desktop', {}).get('page', '')
        else:
            extract, url = '', ''
        items.append({
            'source': 'Wikipedia',
            'title': title,
            'url': url,
            'content': extract or title,
            'timestamp': c.get('timestamp')
        })
    return items

def build_graph(docs: List[Dict]) -> nx.Graph:
    G = nx.Graph()
    for doc in docs:
        keys = extract_keywords(doc.get('title','') + '. ' + doc.get('content',''))
        art_node = f"ART::{doc.get('title','')[:60]}"
        G.add_node(art_node, type='article')
        for k in set(keys):
            kw_node = f"KW::{k}"
            G.add_node(kw_node, type='keyword')
            G.add_edge(art_node, kw_node, weight=1)
    return G

def render_pyvis_graph(G: nx.Graph) -> None:
    nt = Network(height="560px", width="100%", notebook=False, heading="")
    nt.barnes_hut()
    for n, attrs in G.nodes(data=True):
        nt.add_node(n, label=n.replace('ART::','').replace('KW::',''), shape='box' if attrs['type']=='article' else 'dot')
    for a, b, attrs in G.edges(data=True):
        nt.add_edge(a, b, value=attrs.get('weight', 1))
    tmp_path = 'graph.html'
    nt.show(tmp_path)
    with open(tmp_path, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=600, scrolling=True)

def main():
    st.title("🛰️ CTO 2030: AI News & Knowledge Graph Explorer")
    with st.sidebar:
        st.session_state['privacy_mode'] = st.toggle("Privacy Mode", value=True)
        sources = st.multiselect("Data Sources", ["HackerNews","Wikipedia"], default=["HackerNews","Wikipedia"])
        batch_size = st.slider("Items per source", 3, 20, 8)
        run = st.button("⚡ RUN PIPELINE", type="primary")
    if run:
        docs = []
        if "HackerNews" in sources:
            docs.extend(fetch_hn_frontpage(batch_size))
        if "Wikipedia" in sources:
            docs.extend(fetch_wikipedia_recent(batch_size))
        if st.session_state.get('privacy_mode', True):
            for d in docs:
                d['content'] = redact_privacy(d['content'])
        if not docs:
            st.error("No documents fetched.")
            return
        st.dataframe(pd.DataFrame(docs)[["source","title","timestamp","url"]])
        G = build_graph(docs)
        render_pyvis_graph(G)
        joined_texts = "\n\n".join(f"{d['title']}. {d['content']}" for d in docs)
        st.write("### Summary")
        st.write(simple_summarize(joined_texts, 4))
        st.write("### Bias Analysis")
        st.json(analyze_bias(joined_texts))
    else:
        st.info("Select sources and click RUN PIPELINE.")

if __name__ == "__main__":
    main()
