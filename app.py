import streamlit as st
from pyvis.network import Network
import wikipedia

# App title
st.set_page_config(page_title="🛰️ CTO 2030: AI News & Knowledge Graph Explorer", layout="wide")
st.title("🛰️ CTO 2030: AI News & Knowledge Graph Explorer")
st.caption("Real-time tech intelligence with AI insights for the next decade.")

# Sidebar input
st.sidebar.header("Select Data Sources")
wiki_enabled = st.sidebar.checkbox("Wikipedia", value=True)
items_per_source = st.sidebar.slider("Items per source", 1, 20, 5)
wiki_query = st.sidebar.text_input("Wikipedia query", "Artificial Intelligence")

# Only run Wikipedia search if enabled
if wiki_enabled and wiki_query.strip():
    try:
        st.subheader(f"Wikipedia Search: {wiki_query}")
        search_results = wikipedia.search(wiki_query, results=items_per_source)
        st.write(search_results)

        # Create network visualization
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        net.add_node(wiki_query, label=wiki_query, color="#ff0000")

        for title in search_results:
            net.add_node(title, label=title, color="#00ff00")
            net.add_edge(wiki_query, title)

        # Save & display the graph
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=550, scrolling=True)

        # -----------------------
        # Real-time node insights
        # -----------------------
        from transformers import pipeline

        generator = pipeline("text2text-generation", model="google/flan-t5-base")

        def describe_nodes(nodes):
            descriptions = []
            for node in nodes:
                prompt = f"Explain in 2 lines what '{node}' means in technology."
                output = generator(prompt, max_length=50, do_sample=False)
                descriptions.append(f"**{node}**: {output[0]['generated_text']}")
            return descriptions

        graph_nodes = [wiki_query] + search_results  # list of all node labels
        st.markdown("---")
        st.subheader("🔍 Real-time Node Insights")
        for desc in describe_nodes(graph_nodes):
            st.write(desc)

    except Exception as e:
        st.error(f"Error fetching Wikipedia data: {e}")
