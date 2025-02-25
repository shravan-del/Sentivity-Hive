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
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import uvicorn

# --- Setup FastAPI ---
app = FastAPI(
    title="Hive API",
    description="API for text classification, embedding, clustering, and summarization.",
    version="1.0"
)

# --- Load API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# --- Ensure API Keys Exist ---
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError("Missing Reddit API credentials.")

# --- Load Models ---
CLASSIFIER_MODEL_PATH = "AutoClassifier.pkl"
VECTORIZER_MODEL_PATH = "AutoVectorizer.pkl"

try:
    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
except Exception as e:
    raise ValueError(f"Failed to load models: {str(e)}")

# --- Set Up Reddit API ---
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent='HiveRedditScraper/1.0',
        check_for_async=False
    )
except Exception as e:
    raise ValueError(f"Failed to initialize Reddit API: {str(e)}")

# --- Define API Request Models ---
class TextRequest(BaseModel):
    text: str

class EmbedTextRequest(BaseModel):
    texts: List[str]

class ClusterTextRequest(BaseModel):
    texts: List[str]

class SummaryRequest(BaseModel):
    texts: List[str]

# --- Root Route ---
@app.get("/")
def home():
    return {"message": "Hive API is running successfully!"}

# --- Fetch Recent Posts from Reddit ---
@app.get("/fetch_reddit")
def fetch_reddit_posts(subreddit: str = Query(..., description="Subreddit name"),
                        limit: int = Query(100, description="Number of posts to fetch")):
    try:
        posts = [post.title for post in reddit.subreddit(subreddit).new(limit=limit)]
        return {"subreddit": subreddit, "posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching posts: {str(e)}")

# --- API Endpoint: Classify Text ---
@app.post("/predict")
def predict_sentiment(req: TextRequest):
    try:
        text_vectorized = vectorizer.transform([req.text])
        prediction = classifier.predict(text_vectorized)[0]
        return {"text": req.text, "prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

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
def embed_texts(req: EmbedTextRequest):
    try:
        processed_texts = [simple_preprocess(t) for t in req.texts]
        embeddings = embed_model.encode(processed_texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

# --- Clustering Function ---
@app.post("/cluster")
def cluster_texts_api(req: ClusterTextRequest):
    try:
        embeddings = embed_model.encode(req.texts)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
        labels = clusterer.fit_predict(embeddings)
        return {"labels": labels.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

# --- OpenAI API Client ---
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Generate Summary ---
@app.post("/generate_summary")
def summarize_cluster(req: SummaryRequest):
    try:
        prompt = f"""\
        Below are text excerpts from a specific cluster:

        {" ".join(req.texts)}

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
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI summarization error: {str(e)}")

# --- Run the FastAPI Server (Only for Local Testing) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
