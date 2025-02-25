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
subprocess.run(["pip", "install", "--upgrade", "streamlit"], check=True)


# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Please set it in Render.")
    st.stop()

if not NEWS_API_KEY:
    st.error("❌ Missing NEWS_API_KEY. Please set it in Render.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# --- Load Models ---
CLASSIFIER_MODEL_PATH = "AutoClassifier.pkl"
VECTORIZER_MODEL_PATH = "AutoVectorizer.pkl"

try:
    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
except Exception as e:
    st.warning(f"⚠️ Warning: Failed to load classification models. {str(e)}")

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
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()
        if "articles" in data:
            return [article["title"] for article in data["articles"][:num_articles]]
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error fetching news: {str(e)}")
    return ["⚠️ No news articles found. Check your API key or try a different query."]

# --- Cluster Headlines ---
def cluster_headlines(headlines):
    if not headlines:
        return {}

    embeddings = embed_model.encode(headlines)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)

    clustered_data = {}
    for headline, label in zip(headlines, labels):
        clustered_data.setdefault(label, []).append(headline)

    return clustered_data

# --- Summarize Clusters using GPT-4 ---
def summarize_clusters(clustered_data):
    summaries = {}

    for cluster_id, headlines in clustered_data.items():
        if cluster_id == -1:  # Ignore noise points
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
st.set_page_config(page_title="Hive - Top Headlines Analysis", layout="wide")
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
