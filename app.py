import os
import re
import requests
import hdbscan
import praw
import openai
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from fastapi.responses import HTMLResponse
import uvicorn

# --- Setup FastAPI ---
app = FastAPI(
    title="Hive API",
    description="API for top headlines analysis and summarization.",
    version="1.0"
)

# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Add your NewsAPI key if using
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# --- Ensure API Keys Exist ---
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError("Missing Reddit API credentials.")

# --- Load Sentence Embedding Model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Root Route ---
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Hive API is running successfully!</h1>"

# --- Fetch News Headlines ---
@app.get("/fetch_news")
def fetch_news(category: str = "general", limit: int = 5):
    try:
        url = f"https://newsapi.org/v2/top-headlines?category={category}&pageSize={limit}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news_data = response.json()

        if "articles" not in news_data:
            raise HTTPException(status_code=500, detail="Failed to fetch news.")

        headlines = [article["title"] for article in news_data["articles"]]
        return headlines

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

# --- Fetch Reddit Headlines ---
@app.get("/fetch_reddit")
def fetch_reddit_posts(subreddit: str = "worldnews", limit: int = 10):
    try:
        posts = [post.title for post in reddit.subreddit(subreddit).hot(limit=limit)]
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Reddit posts: {str(e)}")

# --- Cluster Headlines ---
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

# --- Summarize Headlines and Return HTML ---
@app.get("/top_headlines", response_class=HTMLResponse)
def summarize_headlines(category: str = "general", limit: int = 10):
    try:
        # Fetch news headlines
        headlines = fetch_news(category, limit)
        clustered_texts = cluster_headlines(headlines)
        summaries = {}

        # Generate Summaries
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

        # --- Format the Output as HTML ---
        html_output = """
        <html>
        <head><title>Hive - Top Headlines Analysis</title></head>
        <body>
            <h1>Hive</h1>
            <h2>Top Headlines Analysis</h2>
        """

        for cluster, summary in summaries.items():
            html_output += f"<h3>{cluster}</h3><ul>"
            for bullet in summary.split("\n"):
                html_output += f"<li>{bullet.strip()}</li>"
            html_output += "</ul>"

        html_output += """
        </body>
        </html>
        """

        return html_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

# --- Run the FastAPI Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
