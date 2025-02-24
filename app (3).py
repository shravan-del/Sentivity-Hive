import os, re, datetime
import numpy as np
import hdbscan, praw, openai
from collections import Counter
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import altair as alt
import base64
import matplotlib
import matplotlib.font_manager as fm
import joblib

# The classifier model file (which outputs 0 or 1)
model = 'AutoClassifier.pkl'


st.set_page_config(page_title="Hive", layout="wide")

# --- Setup API Keys from secrets ---
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent='MyAPI/0.0.1',
    check_for_async=False
)

# --- App Title ---
st.title("Hive")

# --- Data Collection ---
subreddits = [
    "ohio",
    "wayofthebern",
    "centrist",
    "jordanpeterson",
    "walkaway",
    "florida",
    "kratom",
    "nashville",
    "southpark",
    "conservatives",
    "progun",
    "howardstern",
    "buffalobills",
    "libertarian",
    "israel",
    "truechristian",
    "northcarolina",
    "actualpublicfreakouts"
]


end_date = datetime.datetime.utcnow().date()
start_date = end_date - datetime.timedelta(days=14)
start_ts = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
end_ts = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())

@st.cache_data(show_spinner=False)
def fetch_posts(subreddit, start, end, max_posts=100):
    posts = []
    # Grab up to 100 new posts from each subreddit
    for post in reddit.subreddit(subreddit).new(limit=100):
        if start <= post.created_utc <= end:
            posts.append(post.title)
        elif post.created_utc < start:
            break
        if len(posts) >= max_posts:
            break
    return posts

texts = []
#with st.spinner("Fetching posts from Reddit..."):
for sub in subreddits:
    texts.extend(fetch_posts(sub, start_ts, end_ts))
#st.write(f"Fetched **{len(texts)}** posts from Reddit.")

# --- Filter Posts using the Classification Model ---
# Load the classifier model (expects raw text input and outputs 0 or 1)
classifier = joblib.load(model)  # Load your classifier
vectorizer = joblib.load("AutoVectorizer.pkl")  # Load your pre-fitted vectorizer
X = vectorizer.transform(texts)  # Transform your texts using the existing vocabulary

# Check if the number of features matches the classifier's expectation.
expected_features = 5000
if X.shape[1] < expected_features:
    import scipy.sparse as sp
    n_missing = expected_features - X.shape[1]
    # Create a sparse matrix of zeros with the needed shape and horizontally stack it.
    X = sp.hstack([X, sp.csr_matrix((X.shape[0], n_missing))])

predictions = classifier.predict(X)
usable_texts = [text for text, pred in zip(texts, predictions) if pred == 1]
#st.write(f"Filtered usable posts: **{len(usable_texts)}**")


# --- Basic Text Preprocessing ---
def simple_preprocess(text):
    # Lowercase, strip, remove non-alphanumeric chars, collapse extra spaces.
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

#with st.spinner("Preprocessing texts..."):
    # Process only the usable posts
processed_texts = [simple_preprocess(t) for t in usable_texts]

# --- Embedding ---
#with st.spinner("Generating embeddings..."):
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(processed_texts)

# --- Clustering ---
with st.spinner("Clustering texts..."):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
    labels = clusterer.fit_predict(embeddings)
    # Reassign clusters with fewer than 5 points to noise (-1)
    counts = Counter(labels)
    labels = np.array([lbl if counts[lbl] >= 5 else -1 for lbl in labels])
    
    # Group texts into clusters using the original usable posts
    clusters = {}
    for text, lbl in zip(usable_texts, labels):
        clusters.setdefault(lbl, []).append(text)

# --- Summarization and Graphing Functions ---
client = openai.OpenAI()

def generate_summary(cluster_texts):
    prompt = f"""\
    Below are text excerpts from a specific cluster:

    {" ".join(cluster_texts)}
    
    Your task:
    - Summarize the main topics in exactly four bullet points.
    - Each bullet point should be on its own line.
    - Do not use asterisks or other characters besides a hyphen (“- ”) for each bullet point.
    - Provide enough depth in each bullet point.
    - Maintain the same bullet-point format every time.
    - You may reference external knowledge if needed to supplement the summaries.
    
    Now, please provide your bullet-pointed summary:
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def generate_header(cluster_texts):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Below are text excerpts from a specific cluster. "
                "Summarize the main topics into a concise news headline, no quotes on the end, just plain text :\n\n" +
                " ".join(cluster_texts)
            )}
        ]
    )
    return completion.choices[0].message.content

def naive_count_proper_nouns(texts_list):
    """
    A naive approach to detect words that begin with a capital letter
    and continue with lowercase letters.
    """
    pattern = re.compile(r'\b[A-Z][a-z]+\b')
    count = 0
    for text in texts_list:
        matches = pattern.findall(text)
        count += len(matches)
    return count

def get_word_frequencies(texts_list, stopwords=None):
    if stopwords is None:
        stopwords = {"the", "and", "this", "that", "with", "from", "for", "was", "were", "are"}
    all_text = " ".join(texts_list).lower()
    words = re.findall(r'\w+', all_text)
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words)

# --- Identify Top Clusters (ignoring noise) ---
proper_counts = {
    cid: naive_count_proper_nouns(txts)
    for cid, txts in clusters.items()
    if cid != -1
}

# --- TEXT FORMAT (Custom Font & Color for Streamlit) ---
font_path = "AfacadFlux-VariableFont_slnt,wght[1].ttf"  # Update with your actual font file path
prop = fm.FontProperties(fname=font_path, size=24)
custom_font_name = prop.get_name()
#custom_color = "#000000"  # Custom color

with open(font_path, "rb") as f:
    font_data = f.read()
encoded_font = base64.b64encode(font_data).decode()

custom_css = f"""
<style>
@font-face {{
    font-family: '{custom_font_name}';
    src: url(data:font/ttf;base64,{encoded_font});
}}
body, h1, h2, h3, h4, h5, h6, p, .streamlit-expanderHeader {{
    font-family: '{custom_font_name}', sans-serif !important;
    ;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if proper_counts:
    top_clusters = sorted(proper_counts, key=proper_counts.get, reverse=True)[:3]
    st.header("Top Headlines Analysis")
    
    for cid in top_clusters:
        st.subheader(f"{generate_header(clusters[cid])}")
        cluster_texts = clusters[cid]
        #with st.spinner("Generating summary..."):
        summary = generate_summary(cluster_texts)

        st.write(summary)
else:
    st.write("No valid clusters found.")






