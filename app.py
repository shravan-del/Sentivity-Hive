import os
import requests
import streamlit as st
import openai
import hdbscan
from sentence_transformers import SentenceTransformer

# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Ensure API Keys Exist ---
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY environment variable.")
    st.stop()
if not NEWS_API_KEY:
    st.error("Missing NEWS_API_KEY environment variable.")
    st.stop()

# --- Load Sentence Embedding Model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Function to Fetch News Headlines ---
def fetch_news(category="general", limit=5):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&pageSize={limit}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()
    return [article["title"] for article in news_data.get("articles", [])]

# --- Function to Cluster Headlines ---
def cluster_headlines(texts):
    embeddings = embed_model.encode(texts)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)
    
    clustered_texts = {}
    for idx, label in enumerate(labels):
        if label not in clustered_texts:
            clustered_texts[label] = []
        clustered_texts[label].append(texts[idx])
    
    return clustered_texts

# --- Function to Summarize Clusters ---
def summarize_headlines(clustered_texts):
    summaries = {}

    for cluster_id, texts in clustered_texts.items():
        if cluster_id == -1:
            continue  # Skip outliers
        
        prompt = f"""\
        Below are news headlines from a trending topic:
        {" ".join(texts)}

        Your task:
        - Summarize the main themes into exactly four bullet points.
        - Each bullet should start with "- " (hyphen and space).
        - Avoid unnecessary fluff; keep it direct and informative.

        Provide the bullet points now:
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional news summarizer."},
                      {"role": "user", "content": prompt}]
        )

        summaries[f"Cluster {cluster_id}"] = response.choices[0].message["content"]

    return summaries

# --- Streamlit UI ---
st.title("Hive - Top Headlines Analysis")
st.markdown("### Select a News Category to Analyze:")

category = st.selectbox("Category", ["general", "politics", "business", "technology", "sports", "health"])
limit = st.slider("Number of Headlines", 5, 20, 10)

if st.button("Analyze Headlines"):
    st.write("Fetching news and processing...")
    
    headlines = fetch_news(category, limit)
    
    if not headlines:
        st.error("No news articles found. Try another category.")
        st.stop()

    clustered_texts = cluster_headlines(headlines)
    summaries = summarize_headlines(clustered_texts)

    st.header("Top Headlines Analysis")

    for cluster, summary in summaries.items():
        st.subheader(cluster)
        for bullet in summary.split("\n"):
            st.write(f"• {bullet.strip()}")

st.write("Running Hive-like news analysis.")
