import os
import subprocess
import re
import requests
import numpy as np
import hdbscan
import joblib
import openai
import streamlit as st
from collections import Counter
from sentence_transformers import SentenceTransformer

# Ensure Streamlit is installed
subprocess.run(["pip", "install", "--upgrade", "streamlit"], check=True)

# Set default port if not provided by Render
PORT = int(os.environ.get("PORT", 8501))

# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Please set it in Render.")
if not NEWS_API_KEY:
    st.error("❌ Missing NEWS_API_KEY. Please set it in Render.")

# --- Load Models ---
CLASSIFIER_MODEL_PATH = "AutoClassifier.pkl"
VECTORIZER_MODEL_PATH = "AutoVectorizer.pkl"

try:
    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Failed to load models: {str(e)}")

# --- Load Embedding Model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Preprocessing Function ---
def simple_preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Fetch News Headlines ---
def fetch_news(query="latest", num_articles=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if "articles" in data:
        return [article["title"] for article in data["articles"][:num_articles]]
    else:
        return ["⚠️ Error fetching news. Check API key or query."]

# --- Cluster Headlines ---
def cluster_headlines(headlines):
    embeddings = embed_model.encode(headlines)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)
    clustered_data = {label: [] for label in set(labels)}
    
    for headline, label in zip(headlines, labels):
        clustered_data[label].append(headline)

    return clustered_data

# --- Summarize Clusters using GPT-4 ---
def summarize_clusters(clustered_data):
    summaries = {}
    
    for cluster_id, headlines in clustered_data.items():
        if cluster_id == -1:  # Noise points, ignore
            continue
        
        prompt = f"""\
        Below are news headlines from a specific topic cluster:

        {" ".join(headlines)}

        Your task:
        - Summarize the main topics in exactly four bullet points.
        - Each bullet point should be on its own line.
        - Do not use asterisks or other characters besides a hyphen (“- ”) for each bullet point.
        - Provide enough depth in each bullet point.

        Now, please provide your bullet-pointed summary:
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a news summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            summaries[cluster_id] = response["choices"][0]["message"]["content"]
        except Exception as e:
            summaries[cluster_id] = f"⚠️ Error summarizing: {str(e)}"

    return summaries

# --- Streamlit UI ---
st.title("📰 Hive - Top Headlines Analysis")

query = st.text_input("🔍 Enter a News Topic:", "technology")
num_articles = st.slider("📰 Number of Articles", 5, 50, 10)

if st.button("Fetch & Analyze"):
    with st.spinner("Fetching news... ⏳"):
        headlines = fetch_news(query, num_articles)
    
    if headlines:
        st.subheader("📌 Retrieved Headlines:")
        for headline in headlines:
            st.markdown(f"- {headline}")

        with st.spinner("Clustering headlines... ⏳"):
            clustered_data = cluster_headlines(headlines)

        with st.spinner("Summarizing clusters... ⏳"):
            summaries = summarize_clusters(clustered_data)

        st.subheader("🧠 Clustered Analysis")
        for cluster_id, summary in summaries.items():
            st.markdown(f"### 🔹 Cluster {cluster_id}")
            st.markdown(summary)

st.info("✅ Hive AI is running successfully.")

# Run Streamlit manually
if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "app.py", "--server.port", str(PORT), "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"])
