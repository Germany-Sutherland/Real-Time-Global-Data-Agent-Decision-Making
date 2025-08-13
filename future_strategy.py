# future_strategy.py
import json
from transformers import pipeline

# Load FMEA data
with open("fmea.json", "r") as f:
    fmea_data = json.load(f)

# Load knowledge graph data
with open("knowledge_graph.json", "r") as f:
    kg_data = json.load(f)

# Sentiment Analysis using a lightweight, free model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
node_text = " ".join(kg_data.get("nodes", []))
sentiments = sentiment_analyzer(node_text)

# Use a small open-source text generation model for strategy
strategy_generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"  # lightweight model for free hosting
)

prompt = (
    "Generate a Future Strategy (80-100 lines) integrating Knowledge Graph insights, summaries, "
    "sentiment analysis results, CTO Kumar AI Agent recommendations, and FMEA analysis. "
    "Ensure the tone is professional and forward-looking."
)
strategy_text = strategy_generator(prompt, max_length=800, num_return_sequences=1)[0]['generated_text']

# Save strategy output
with open("future_strategy.txt", "w") as f:
    f.write(strategy_text)

print("✅ Future Strategy generated and saved to future_strategy.txt")
