import os
import re
import requests
import numpy as np
import hdbscan
import joblib
import openai
import streamlit as st
from collections import Counter
from sentence_transformers import SentenceTransformer

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

# --- Apply Custom CSS ---
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: Arial, sans-serif;
    }
    .title {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        color: #ffcc00;
        margin-top: 20px;
    }
    .headline {
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    .summary {
        font-size: 16px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.markdown('<div class="title">📰 Hive - Top Headlines Analysis</div>', unsafe_allow_html=True)

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
query = st.text_input("🔍 Enter a News Topic:", "technology")
num_articles = st.slider("📰 Number of Articles", 5, 50, 10)

if st.button("Fetch & Analyze"):
    with st.spinner("Fetching news... ⏳"):
        headlines = fetch_news(query, num_articles)
    
    if headlines:
        st.markdown('<div class="subheader">📌 Retrieved Headlines:</div>', unsafe_allow_html=True)
        for headline in headlines:
            st.markdown(f'<div class="headline">🔹 {headline}</div>', unsafe_allow_html=True)

        with st.spinner("Clustering headlines... ⏳"):
            clustered_data = cluster_headlines(headlines)

        with st.spinner("Summarizing clusters... ⏳"):
            summaries = summarize_clusters(clustered_data)

        st.markdown('<div class="subheader">🧠 Clustered Analysis</div>', unsafe_allow_html=True)
        for cluster_id, summary in summaries.items():
            st.markdown(f'<div class="headline">📌 Cluster {cluster_id}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary">{summary}</div>', unsafe_allow_html=True)

st.success("✅ Hive AI is running successfully.")
