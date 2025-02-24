import os
import re
import datetime
import numpy as np
import hdbscan
import praw
import openai
import joblib
import json
from collections import Counter
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# --- Setup FastAPI ---
app = FastAPI()

# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# --- Verify API Keys ---
if not OPENAI_API_KEY or not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError("Missing environment variables. Set OPENAI_API_KEY, REDDIT_CLIENT_ID, and REDDIT_CLIENT_SECRET.")

# --- Load Classifier & Vectorizer ---
CLASSIFIER_MODEL_PATH = "AutoClassifier.pkl"
VECTORIZER_MODEL_PATH = "AutoVectorizer.pkl"

classifier = joblib.load(CLASSIFIER_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_MODEL_PATH)

# --- Set Up Reddit API ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent='HiveRedditScraper/1.0',
    check_for_async=False
)

# --- Define Pydantic Model for API ---
class RedditRequest(BaseModel):
    subreddit: str
    limit: int = 100

class TextRequest(BaseModel):
    text: str

# --- Subreddits to Track ---
subreddits = [
    "ohio", "wayofthebern", "centrist", "jordanpeterson", "walkaway",
    "florida", "kratom", "nashville", "southpark", "conservatives",
    "progun", "howardstern", "buffalobills", "libertarian", "israel",
    "truechristian", "northcarolina", "actualpublicfreakouts"
]

# --- Fetch Recent Posts from Reddit ---
def fetch_posts(subreddit: str, limit: int = 100):
    posts = []
    try:
        for post in reddit.subreddit(subreddit).new(limit=limit):
            posts.append(post.title)
    except Exception as e:
        return {"error": f"Failed to fetch posts: {str(e)}"}
    return posts

# --- API Endpoint: Fetch Posts from a Subreddit ---
@app.get("/fetch_reddit")
def fetch_reddit_posts(req: RedditRequest):
    posts = fetch_posts(req.subreddit, req.limit)
    return {"subreddit": req.subreddit, "posts": posts}

# --- API Endpoint: Classify Text ---
@app.post("/predict")
def predict_sentiment(req: TextRequest):
    try:
        text_vectorized = vectorizer.transform([req.text])
        prediction = classifier.predict(text_vectorized)[0]
        return {"text": req.text, "prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

# --- Preprocessing Function ---
def simple_preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Sentence Embedding Model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- API Endpoint: Generate Text Embeddings ---
@app.post("/embed_texts")
def embed_texts(texts: List[str]):
    processed_texts = [simple_preprocess(t) for t in texts]
    embeddings = embed_model.encode(processed_texts)
    return {"embeddings": embeddings.tolist()}

# --- Clustering Function ---
def cluster_texts(texts: List[str]):
    embeddings = embed_model.encode(texts)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
    labels = clusterer.fit_predict(embeddings)
    return {"labels": labels.tolist()}

# --- API Endpoint: Cluster Texts ---
@app.post("/cluster")
def cluster_texts_api(req: List[str]):
    return cluster_texts(req)

# --- OpenAI Summary Functions ---
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_summary(cluster_texts: List[str]):
    prompt = f"""\
    Below are text excerpts from a specific cluster:

    {" ".join(cluster_texts)}

    Your task:
    - Summarize the main topics in exactly four bullet points.
    - Each bullet point should be on its own line.
    - Do not use asterisks or other characters besides a hyphen (“- ”) for each bullet point.
    - Provide enough depth in each bullet point.

    Now, please provide your bullet-pointed summary:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- API Endpoint: Generate Summary from Clustered Texts ---
@app.post("/generate_summary")
def summarize_cluster(req: List[str]):
    return {"summary": generate_summary(req)}

# --- Run the FastAPI Server (Only for Local Testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
