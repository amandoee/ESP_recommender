"""
Jester Joke Recommender System
Unified script with Flask web app + standalone analysis capabilities.

Supports 4 recommendation algorithms:
1. Popularity-based (baseline)
2. Content-based (TF-IDF + cosine similarity)
3. Collaborative filtering (user-based k-NN)
4. Deep learning (two-tower embeddings)

Can run as standalone analysis script or Flask web app.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from datetime import datetime
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

EVALUATION_CACHE_VERSION = 6

# ==========================================
# 0. DATA LOADING AND PREPARATION
# ==========================================

def load_jester_dataset(use_cache=True, cache_dir=".cache"):
    """
    Load the Jester dataset from Excel files.
    
    Format:
    - Each ratings file is a matrix: rows=users, columns=joke info + ratings
    - Column 0: Count of jokes rated by that user
    - Columns 1-100: Ratings for jokes 01-100
    - Ratings range from -10 to +10 (99 = not rated/null)
    - User IDs are implicit (row indices)
    
    Returns:
        jokes_df: DataFrame with columns 'joke_id', 'joke_text'
        ratings_df: DataFrame with columns 'user_id', 'joke_id', 'rating'
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    cache_path = Path(base_path) / cache_dir
    jokes_cache = cache_path / "jokes_df.pkl"
    ratings_cache = cache_path / "ratings_df.pkl"

    if use_cache and jokes_cache.exists() and ratings_cache.exists():
        print("Loading datasets from cache...")
        jokes_df = pd.read_pickle(jokes_cache)
        ratings_df = pd.read_pickle(ratings_cache)
        print(f"Loaded {len(jokes_df)} jokes and {len(ratings_df)} ratings from cache")
        return jokes_df, ratings_df
    
    # Load joke dataset from xlsx
    print("Loading jokes dataset...")
    jokes_file = os.path.join(base_path, "Dataset4JokeSet", "Dataset4JokeSet.xlsx")
    jokes_raw = pd.read_excel(jokes_file)
    
    # Extract joke texts - the file has jokes as header and in first column
    jokes_texts = []
    first_col = jokes_raw.columns[0]
    jokes_texts.append(first_col)
    
    for val in jokes_raw.iloc[:, 0]:
        jokes_texts.append(val)
    
    jokes_df = pd.DataFrame({
        'joke_id': range(len(jokes_texts)),
        'joke_text': jokes_texts
    })
    print(f"Loaded {len(jokes_df)} jokes")
    
    # Load ratings from three datasets
    print("Loading ratings datasets...")
    ratings_files = [
        os.path.join(base_path, "jester_dataset_1_1", "jester-data-1.xls"),
        os.path.join(base_path, "jester_dataset_1_2", "jester-data-2.xls"),
        os.path.join(base_path, "jester_dataset_1_3", "jester-data-3.xls"),
    ]
    
    ratings_list = []
    total_users = 0
    
    for file_idx, file in enumerate(ratings_files):
        if os.path.exists(file):
            print(f"  Loading {os.path.basename(file)}...")
            df = pd.read_excel(file, engine='xlrd', header=None)
            
            # Column 0 is count of rated jokes, columns 1-100 are joke ratings.
            # Vectorized conversion is much faster than nested Python loops.
            num_users = df.shape[0]
            ratings_matrix = df.iloc[:, 1:101].copy()
            ratings_matrix.columns = range(100)

            df_long = ratings_matrix.stack().reset_index()
            df_long.columns = ["local_user_id", "joke_id", "rating"]
            df_long["user_id"] = df_long["local_user_id"] + total_users
            df_long = df_long[["user_id", "joke_id", "rating"]]

            ratings_list.append(df_long)
            total_users += num_users
    
    if ratings_list:
        ratings_df = pd.concat(ratings_list, ignore_index=True)
        print(f"Loaded {len(ratings_df)} ratings from {total_users} users")
    else:
        print("Warning: No ratings files found")
        ratings_df = pd.DataFrame()
    
    # Clean: replace 99 (missing values) with NaN
    if 'rating' in ratings_df.columns:
        ratings_df['rating'] = ratings_df['rating'].replace(99, np.nan)

    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)
        jokes_df.to_pickle(jokes_cache)
        ratings_df.to_pickle(ratings_cache)
        print(f"Saved dataset cache to {cache_path}")
    
    return jokes_df, ratings_df


def prepare_data_for_models(jokes_df, ratings_df):
    """
    Prepare and validate data for use in recommendation models.
    
    Returns:
        jokes_df: DataFrame with 'joke_id', 'joke_text'
        ratings_df: DataFrame with 'user_id', 'joke_id', 'rating' (cleaned)
    """
    print("\n--- DATA PREPARATION ---")
    print(f"Jokes: {len(jokes_df)} records")
    print(f"Ratings: {len(ratings_df)} records")
    
    # Remove missing ratings
    if 'rating' in ratings_df.columns:
        ratings_clean = ratings_df.dropna(subset=['rating'])
        removed = len(ratings_df) - len(ratings_clean)
        print(f"Removed {removed} ratings with missing values")
        ratings_df = ratings_clean
    
    return jokes_df, ratings_df


# ==========================================
# 1. BASELINE: POPULARITY MODEL
# ==========================================

def popularity_recommender(jokes_df, ratings_df, top_n=5):
    """Recommend jokes based on average rating from your loaded data"""
    print("--- POPULARITY MODEL ---")
    
    # Calculate average rating per joke from your ratings data
    avg_ratings = ratings_df.groupby('joke_id')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['joke_id', 'avg_rating', 'num_ratings']
    
    # Merge with jokes to get full info
    joke_scores = jokes_df.merge(avg_ratings, on='joke_id', how='left')
    joke_scores = joke_scores.dropna(subset=['avg_rating'])
    
    # Sort by average rating
    top_jokes = joke_scores.nlargest(top_n, 'avg_rating')
    
    print(f"Top {top_n} Most Popular Jokes:")
    for idx, row in top_jokes.iterrows():
        print(f"  Joke {row['joke_id']}: Rating {row['avg_rating']:.2f} ({int(row['num_ratings'])} ratings)")
        print(f"    {row['joke_text'][:80]}...\n")
    
    print(f"EXPLAINABILITY: We recommend these jokes because they have the highest average ratings across all users.\n")


# ==========================================
# 2. CONTENT-BASED FILTERING
# ==========================================

def content_based_recommender(jokes_df, target_joke_id, top_n=5):
    """Recommend jokes based on text similarity using your loaded jokes data"""
    print(f"--- CONTENT-BASED FILTERING (Similar to Joke {target_joke_id}) ---")
    
    if target_joke_id not in jokes_df['joke_id'].values:
        print(f"Error: Joke {target_joke_id} not found")
        return
    
    # TF-IDF vectorizer extracts keywords from joke text
    tfv = TfidfVectorizer(stop_words='english', max_features=100)
    jokes_df_copy = jokes_df.copy()
    jokes_df_copy['joke_text'] = jokes_df_copy['joke_text'].fillna('')
    tfv_matrix = tfv.fit_transform(jokes_df_copy['joke_text'])
    
    # Calculate Cosine Similarity between all jokes
    cosine_sim = linear_kernel(tfv_matrix, tfv_matrix)
    
    # Get the index of target joke
    target_idx = jokes_df_copy[jokes_df_copy['joke_id'] == target_joke_id].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar jokes (skip first one which is the target joke itself)
    top_similar_indices = [sim_scores[i][0] for i in range(1, min(top_n + 1, len(sim_scores)))]
    
    print(f"Top {top_n} Similar Jokes:")
    for idx in top_similar_indices:
        joke_id = jokes_df_copy.iloc[idx]['joke_id']
        print(f"  Joke {int(joke_id)}: {jokes_df_copy.iloc[idx]['joke_text'][:80]}...\n")
    
    print(f"EXPLAINABILITY: We recommend these because their text content is mathematically similar to the joke you selected.\n")


# ==========================================
# 3. MEMORY-BASED COLLABORATIVE FILTERING
# ==========================================

def memory_based_cf(ratings_df, min_ratings_per_user=5):
    """User-based collaborative filtering using your loaded ratings data
    
    Args:
        ratings_df: DataFrame with user, joke, and rating columns
        min_ratings_per_user: Minimum ratings per user to avoid sparsity issues (default 5)
    """
    print("--- MEMORY-BASED COLLABORATIVE FILTERING (User-Based k-NN) ---")
    
    # Filter users with at least min_ratings_per_user ratings to avoid sparsity
    user_counts = ratings_df.groupby('user_id').size()
    users_with_enough_ratings = user_counts[user_counts >= min_ratings_per_user].index
    ratings_filtered = ratings_df[ratings_df['user_id'].isin(users_with_enough_ratings)].copy()
    
    print(f"Filtered to users with ≥{min_ratings_per_user} ratings: {len(users_with_enough_ratings)} users")
    
    print(f"Using {len(users_with_enough_ratings)} users for training")

    user_item = ratings_filtered.pivot_table(
        index='user_id',
        columns='joke_id',
        values='rating',
        aggfunc='mean'
    ).reindex(columns=range(100))

    joke_means = user_item.mean(axis=0)
    user_item_filled = user_item.fillna(joke_means).fillna(0.0)

    k_neighbors = min(40, len(user_item_filled))
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(user_item_filled.values)

    print("Training user-based k-NN model on all eligible users complete.")
    print("EXPLAINABILITY: We recommend jokes based on users with similar rating patterns to you.\n")

    return nn_model


# ==========================================
# 4. DEEP LEARNING (TENSORFLOW RECOMMENDERS)
# ==========================================

def deep_learning_recommender(ratings_df, jokes_df):
    """Deep learning two-tower model using your loaded data"""
    print("--- DEEP LEARNING MODEL (Two-Tower Retrieval) ---")
    
    try:
        import tensorflow as tf
        import tensorflow_recommenders as tfrs
    except (ImportError, RecursionError) as e:
        print(f"Error: tensorflow_recommenders not available or has compatibility issues.")
        print(f"Details: {type(e).__name__}: {str(e)[:100]}")
        print(f"Skipping deep learning model. Other recommendation methods will still work.")
        return None
    
    print("Preparing data for deep learning model...")
    ratings_df_copy = ratings_df.dropna(subset=['rating']).copy()
    ratings_df_copy['user_id'] = ratings_df_copy['user_id'].astype(str)
    ratings_df_copy['joke_id'] = ratings_df_copy['joke_id'].astype(str)

    ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df_copy[['user_id', 'joke_id']]))
    jokes = tf.data.Dataset.from_tensor_slices(jokes_df['joke_id'].astype(str).values)

    # Build the Model
    class JesterModel(tfrs.Model):
        def __init__(self, num_users, num_items):
            super().__init__()
            self.user_model = tf.keras.layers.Embedding(input_dim=num_users, output_dim=64)
            self.item_model = tf.keras.layers.Embedding(input_dim=num_items, output_dim=64)
            
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=jokes.batch(128).map(self.item_model)
                )
            )

        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            joke_embeddings = self.item_model(features["joke_id"])
            return self.task(user_embeddings, joke_embeddings)

    num_users = int(ratings_df_copy['user_id'].astype(int).max()) + 1
    num_items = int(ratings_df_copy['joke_id'].astype(int).max()) + 1
    
    model = JesterModel(num_users, num_items)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # Shuffle and split
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    train_size = int(len(ratings_df_copy) * 0.8)
    train = shuffled.take(train_size)
    test = shuffled.skip(train_size)

    # Train
    print("Training model...")
    model.fit(train.batch(2048), epochs=3, verbose=1)
    
    print("Evaluating model...")
    train_eval = model.evaluate(train.batch(2048), return_dict=True, verbose=0)
    test_eval = model.evaluate(test.batch(2048), return_dict=True, verbose=0)
    
    print(f"TRAIN METRICS: {train_eval}")
    print(f"TEST METRICS: {test_eval}")
    print(f"EXPLAINABILITY: Low. Deep learning models learn complex latent factors from user-item interactions.\n")

    # Extract item embeddings as a numpy array for serving (picklable)
    item_ids_tensor = tf.constant(list(range(num_items)), dtype=tf.int32)
    item_embeddings = model.item_model(item_ids_tensor).numpy()  # shape: [num_items, 64]
    return model, item_embeddings


# ==========================================
# FLASK APP INITIALIZATION
# ==========================================

app = Flask(__name__)

# Global data and precomputed artifacts
jokes_df = None
ratings_df = None
ARTIFACTS = {}


def _cache_paths():
    base = Path(__file__).resolve().parent / ".cache"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "precomputed": base / "precomputed_artifacts.pkl",
        "evaluation": base / "evaluation_results.pkl",
    }


def _is_cached_evaluation_current(cached_results):
    return (
        isinstance(cached_results, dict)
        and cached_results.get("cache_version") == EVALUATION_CACHE_VERSION
        and cached_results.get("metric_name") == "Recall@10"
        and cached_results.get("top_metric_name") == "Top@10"
    )


def _build_artifacts(jokes, ratings):
    print("Precomputing artifacts...")

    # 1) Popularity ranking (computed once).
    avg = ratings.groupby("joke_id")["rating"].agg(["mean", "count"]).reset_index()
    avg.columns = ["joke_id", "avg_rating", "num_ratings"]
    popularity_table = avg.sort_values(["avg_rating", "num_ratings"], ascending=False).reset_index(drop=True)

    # 2) Content-based matrices (fit once).
    tfidf = TfidfVectorizer(stop_words="english", max_features=300)
    texts = jokes["joke_text"].fillna("")
    tfidf_matrix = tfidf.fit_transform(texts)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # 3) User-based collaborative artifacts (train NearestNeighbors once).
    user_counts = ratings.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 5].index.to_numpy()
    sampled = ratings[ratings["user_id"].isin(eligible_users)].copy()

    print(f"Training user-based k-NN model on all {len(eligible_users)} eligible users...")
    
    # Build user-item matrix for similarity-based predictions
    user_item = sampled.pivot_table(index="user_id", columns="joke_id", values="rating", aggfunc="mean")
    user_item = user_item.reindex(columns=range(100))
    joke_means = user_item.mean(axis=0)
    user_item_filled = user_item.fillna(joke_means).fillna(0.0)

    k_neighbors = min(40, len(user_item_filled))
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn_model.fit(user_item_filled.values)

    # 4) Deep learning (two-tower) artifact via TF Recommenders fit once.
    dl_result = deep_learning_recommender(ratings, jokes)
    item_embeddings = dl_result[1] if dl_result is not None else None

    return {
        "popularity_table": popularity_table,
        "candidate_joke_ids": set(popularity_table["joke_id"].astype(int).tolist()),
        "tfidf": tfidf,
        "cosine_sim": cosine_sim,
        "user_item": user_item,
        "user_item_filled": user_item_filled,
        "joke_means": joke_means,
        "nn_model": nn_model,
        "item_embeddings": item_embeddings,
    }


def init_app():
    """Load data and load/build precomputed artifacts once."""
    global jokes_df, ratings_df, ARTIFACTS

    print("Loading datasets...")
    jokes_df, ratings_df = load_jester_dataset(use_cache=True)
    jokes_df, ratings_df = prepare_data_for_models(jokes_df, ratings_df)

    cache_paths = _cache_paths()
    cache_file = cache_paths["precomputed"]

    if cache_file.exists():
        print("Loading precomputed artifacts from cache...")
        with cache_file.open("rb") as f:
            ARTIFACTS = pickle.load(f)
        required_keys = {"popularity_table", "candidate_joke_ids", "cosine_sim", "user_item", "user_item_filled", "joke_means", "nn_model", "item_embeddings"}
        if not required_keys.issubset(set(ARTIFACTS.keys())):
            print("Cached artifacts are outdated. Rebuilding...")
            ARTIFACTS = _build_artifacts(jokes_df, ratings_df)
            with cache_file.open("wb") as f:
                pickle.dump(ARTIFACTS, f)
    else:
        ARTIFACTS = _build_artifacts(jokes_df, ratings_df)
        with cache_file.open("wb") as f:
            pickle.dump(ARTIFACTS, f)

    ensure_evaluation_results_cached()

    print("App ready: precomputations loaded.")

@app.route('/')
def index():
    """Home page - display rating interface"""
    # Very cheap at runtime: only random sampling from in-memory frame.
    candidates = jokes_df[jokes_df["joke_id"].isin(ARTIFACTS["candidate_joke_ids"])]
    random_jokes = candidates.sample(n=10, random_state=None).reset_index(drop=True)
    jokes_list = random_jokes[['joke_id', 'joke_text']].to_dict('records')
    
    return render_template('index.html', jokes=jokes_list)

@app.route('/get-recommendations', methods=['POST'])
def get_recommendations():
    """
    Receive user ratings and return recommendations from all 4 methods
    """
    try:
        data = request.get_json()
        user_ratings = data.get('ratings', {})  # {joke_id: rating_value}
        
        if not user_ratings:
            return jsonify({'error': 'No ratings provided'}), 400
        
        candidate_jokes = ARTIFACTS["candidate_joke_ids"]
        rated_jokes = set(int(jid) for jid in user_ratings.keys() if int(jid) in candidate_jokes)
        if not rated_jokes:
            return jsonify({'error': 'No valid ratings for supported joke IDs'}), 400

        user_vector = pd.Series(index=range(100), dtype=float)
        for jid, rating in user_ratings.items():
            jid_int = int(jid)
            if jid_int in candidate_jokes:
                user_vector[jid_int] = float(rating)
        
        recommendations = {}
        
        # 1. POPULARITY-BASED RECOMMENDATIONS (not user-based)
        try:
            popularity_table = ARTIFACTS["popularity_table"]
            top_3 = popularity_table[~popularity_table["joke_id"].isin(rated_jokes)].head(3)
            recommendations['popularity'] = [
                {
                    'joke_id': int(row['joke_id']),
                    'joke_text': jokes_df[jokes_df['joke_id'] == row['joke_id']]['joke_text'].values[0],
                    'avg_rating': round(float(row['avg_rating']), 2),
                    'num_ratings': int(row['num_ratings']),
                    'reasoning': f"This joke has the highest average rating ({round(float(row['avg_rating']), 2)}) among all users ({int(row['num_ratings'])} ratings). Popularity-based recommendations show jokes that most people enjoy."
                }
                for _, row in top_3.iterrows()
            ]
        except Exception as e:
            recommendations['popularity'] = [{'error': str(e)}]
        
        # 2. CONTENT-BASED RECOMMENDATIONS (weighted by user ratings)
        try:
            cosine_sim = ARTIFACTS["cosine_sim"]
            rated_indices = [jokes_df[jokes_df['joke_id'] == int(jid)].index[0] for jid in user_ratings.keys()]
            
            # Weight similarities by user ratings: normalize ratings to 0-1 range
            # -10 → 0, 0 → 0.5, +10 → 1 (using formula: (rating + 10) / 20)
            ratings_values = np.array([float(user_ratings[str(jokes_df.iloc[idx]['joke_id'])]) for idx in rated_indices])
            weights = (ratings_values + 10) / 20  # Normalize to 0-1 range
            weights = weights / weights.sum()  # Ensure weights sum to 1 for proper averaging
            
            # Calculate weighted average similarity
            weighted_sims = np.average(cosine_sim[rated_indices], axis=0, weights=weights)
            ranked_jokes = np.argsort(weighted_sims)[::-1]
            top_3_indices = [
                idx
                for idx in ranked_jokes
                if jokes_df.iloc[idx]['joke_id'] in candidate_jokes and jokes_df.iloc[idx]['joke_id'] not in rated_jokes
            ][:3]
            
            recommendations['content_based'] = [
                {
                    'joke_id': int(jokes_df.iloc[idx]['joke_id']),
                    'joke_text': jokes_df.iloc[idx]['joke_text'],
                    'similarity_score': round(float(weighted_sims[idx]), 4),
                    'reasoning': f"This joke's text is highly similar (similarity: {round(float(weighted_sims[idx]), 4)}) to jokes you rated highly. Jokes with similar wording/topics tend to appeal to the same sense of humor."
                }
                for idx in top_3_indices
            ]
        except Exception as e:
            recommendations['content_based'] = [{'error': str(e)}]
        
        # 3. COLLABORATIVE FILTERING RECOMMENDATIONS (using sklearn NearestNeighbors)
        try:
            user_item = ARTIFACTS["user_item"]
            user_item_filled = ARTIFACTS["user_item_filled"]
            nn_model = ARTIFACTS["nn_model"]
            joke_means = ARTIFACTS["joke_means"]

            # For new users, compute similarity to training set users using cosine similarity
            filled_user = user_vector.copy()
            for jid in range(100):
                if pd.isna(filled_user.loc[jid]):
                    filled_user.loc[jid] = joke_means.loc[jid] if not pd.isna(joke_means.loc[jid]) else 0.0

            _, indices = nn_model.kneighbors([filled_user.values], return_distance=True)
            neighbor_ids = user_item_filled.index[indices[0]]

            neighbor_ratings = user_item.loc[neighbor_ids]
            unrated_jokes = candidate_jokes - rated_jokes
            predictions = []
            for joke_id in unrated_jokes:
                pred = neighbor_ratings[joke_id].mean(skipna=True)
                if pd.isna(pred):
                    pred = joke_means.loc[joke_id] if not pd.isna(joke_means.loc[joke_id]) else 0.0
                predictions.append({
                    'joke_id': int(joke_id),
                    'pred_rating': float(pred)
                })
            
            top_3_cf = sorted(predictions, key=lambda x: x['pred_rating'], reverse=True)[:3]
            recommendations['collaborative'] = [
                {
                    'joke_id': rec['joke_id'],
                    'joke_text': jokes_df[jokes_df['joke_id'] == rec['joke_id']]['joke_text'].values[0],
                    'predicted_rating': round(rec['pred_rating'], 2),
                    'reasoning': f"Users with rating patterns similar to yours enjoyed this joke (predicted rating: {round(rec['pred_rating'], 2)}). This method finds people like you and recommends what they liked."
                }
                for rec in top_3_cf
            ]
        except Exception as e:
            recommendations['collaborative'] = [{'error': str(e)}]
        
        # 4. DEEP LEARNING RECOMMENDATIONS
        try:
            item_embeddings = ARTIFACTS["item_embeddings"]
            if item_embeddings is None:
                raise ValueError("Deep learning model unavailable (TensorFlow/TF-Recommenders not installed)")

            # Cold-start: represent the new user as a weighted average of rated item embeddings
            rated_embeddings = [
                item_embeddings[int(jid)] * float(rating)
                for jid, rating in user_ratings.items()
                if int(jid) in candidate_jokes
            ]
            user_embedding = np.mean(rated_embeddings, axis=0)

            unrated_jokes = candidate_jokes - rated_jokes
            predictions_dl = []
            for joke_id in unrated_jokes:
                score = float(np.dot(user_embedding, item_embeddings[joke_id]))
                predictions_dl.append({'joke_id': int(joke_id), 'pred_rating': score})

            max_score = max((p['pred_rating'] for p in predictions_dl), default=1.0)
            top_3_dl = sorted(predictions_dl, key=lambda x: x['pred_rating'], reverse=True)[:3]
            recommendations['deep_learning'] = [
                {
                    'joke_id': rec['joke_id'],
                    'joke_text': jokes_df[jokes_df['joke_id'] == rec['joke_id']]['joke_text'].values[0],
                    'confidence': round(max(0, rec['pred_rating']) / max(1e-9, abs(max_score)) * 100, 1),
                    'reasoning': f"This joke matches hidden patterns learned from your rating behavior (confidence: {round(max(0, rec['pred_rating']) / max(1e-9, abs(max_score)) * 100, 1)}%). Deep learning captures complex preferences beyond simple similarity."
                }
                for rec in top_3_dl
            ]
        except Exception as e:
            recommendations['deep_learning'] = [{'error': str(e)}]
        
        return jsonify(recommendations)
    
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# STANDALONE ANALYSIS MODE
# ==========================================

def main():
    """Run as standalone analysis script"""
    # Load the datasets
    jokes_df_local, ratings_df_local = load_jester_dataset()
    
    # Prepare data for models
    jokes_df_local, ratings_df_local = prepare_data_for_models(jokes_df_local, ratings_df_local)
    
    # Inspect the data structure
    print("\n--- DATA STRUCTURE ---")
    print(f"\nJokes DataFrame:")
    print(jokes_df_local.head())
    print(f"\nRatings DataFrame:")
    print(ratings_df_local.head())
    
    # Run the recommender models with your loaded data
    print("\n" + "="*60)
    popularity_recommender(jokes_df_local, ratings_df_local, top_n=5)
    
    print("="*60)
    content_based_recommender(jokes_df_local, target_joke_id=0, top_n=5)
    
    print("="*60)
    memory_based_cf(ratings_df_local)


def evaluate_recommenders(
    jokes_df_eval,
    ratings_df_eval,
    test_size=0.3,
    random_state=42,
    top_n=10,
    relevance_threshold=0.0,
    include_deep_learning=True,
    verbose=True,
):
    """Evaluate recommenders with per-user joke holdout Recall@10 and Top@10 on test ranking."""
    if verbose:
        print("\n" + "=" * 80)
        print(f"OBJECTIVE EVALUATION (PER-USER JOKE HOLDOUT RECALL@{top_n} + TOP@{top_n})")
        print("=" * 80)

    ratings_clean = ratings_df_eval.dropna(subset=["rating"]).copy()
    joke_ids = np.sort(ratings_clean["joke_id"].astype(int).unique())
    jokes_sorted = (
        jokes_df_eval[jokes_df_eval["joke_id"].isin(joke_ids)]
        .sort_values("joke_id")
        .reset_index(drop=True)
    )
    item_to_col = {int(jid): idx for idx, jid in enumerate(joke_ids)}

    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    users_considered = 0
    users_skipped_for_split = 0

    for user_id, group in ratings_clean.groupby("user_id", sort=True):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        total_rated = len(group)
        if total_rated < 2:
            users_skipped_for_split += 1
            continue

        test_count = max(1, int(np.ceil(total_rated * test_size)))
        if test_count >= total_rated:
            test_count = total_rated - 1

        train_count = total_rated - test_count
        if train_count < 1:
            users_skipped_for_split += 1
            continue

        train_parts.append(group.iloc[:train_count].copy())
        test_parts.append(group.iloc[train_count:].copy())
        users_considered += 1

    if not train_parts or not test_parts:
        raise ValueError("Not enough ratings to build a per-user joke holdout evaluation split")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    if verbose:
        print(f"Train size: {len(train_df):,} ratings ({(1 - test_size) * 100:.0f}%)")
        print(f"Test size:  {len(test_df):,} ratings ({test_size * 100:.0f}%)")
        print(f"Users with valid train/test holdout: {users_considered:,}")
        if users_skipped_for_split:
            print(f"Users skipped during split: {users_skipped_for_split:,}")

    global_mean = float(train_df["rating"].mean())
    train_joke_means = train_df.groupby("joke_id")["rating"].mean()

    test_by_user = {
        uid: grp[["joke_id", "rating"]].copy().reset_index(drop=True)
        for uid, grp in test_df.groupby("user_id", sort=False)
    }
    train_seen_by_user = {
        uid: set(grp["joke_id"].astype(int).tolist())
        for uid, grp in train_df.groupby("user_id", sort=False)
    }
    eligible_eval_users = []
    relevant_test_by_user = {}
    for uid, test_user in test_by_user.items():
        relevant = set(test_user.loc[test_user["rating"] > relevance_threshold, "joke_id"].astype(int).tolist())
        if relevant:
            eligible_eval_users.append(uid)
            relevant_test_by_user[uid] = relevant

    if not eligible_eval_users:
        raise ValueError("No users have relevant held-out test jokes for Recall@10 evaluation")

    top_eval_users = [uid for uid, user_test in test_by_user.items() if len(user_test) > 0]

    results = []

    def _compute_metrics(name, ranked_lists):
        recalls = []
        top_hits = []

        for uid in eligible_eval_users:
            ranked_items = ranked_lists.get(uid, [])
            predicted_top = ranked_items[:top_n]
            relevant = relevant_test_by_user[uid]
            hit_count = len(set(predicted_top) & relevant)
            recalls.append(hit_count / len(relevant))

        for uid in top_eval_users:
            user_test = test_by_user[uid]
            test_jokes = set(user_test["joke_id"].astype(int).tolist())
            if not test_jokes:
                continue

            actual_top = (
                user_test.sort_values(["rating", "joke_id"], ascending=[False, True])["joke_id"]
                .astype(int)
                .tolist()[:top_n]
            )
            if not actual_top:
                continue

            ranked_items = ranked_lists.get(uid, [])
            predicted_test_ranked = [int(jid) for jid in ranked_items if int(jid) in test_jokes]
            predicted_top = predicted_test_ranked[:top_n]

            top_hit = len(set(predicted_top) & set(actual_top)) / len(actual_top)
            top_hits.append(top_hit)

        recall_score = float(np.mean(recalls)) if recalls else 0.0
        top_score = float(np.mean(top_hits)) if top_hits else 0.0
        results.append((name, recall_score, top_score))
        if verbose:
            print(f"{name:<20} Recall@{top_n}={recall_score:.4f} | Top@{top_n}={top_score:.4f}")

    def _sorted_unseen_items(score_map, seen_jokes):
        return [
            joke_id
            for joke_id, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            if joke_id not in seen_jokes
        ]

    # 1) Popularity baseline
    popularity_scores = {int(jid): float(train_joke_means.get(jid, global_mean)) for jid in joke_ids}
    popularity_ranked = {
        uid: _sorted_unseen_items(popularity_scores, train_seen_by_user[uid])
        for uid in eligible_eval_users
    }
    _compute_metrics("Popularity", popularity_ranked)

    # 1b) Random top-N baseline
    random_ranked = {}
    for uid in eligible_eval_users:
        seen_jokes = train_seen_by_user[uid]
        unseen_jokes = [int(joke_id) for joke_id in joke_ids if int(joke_id) not in seen_jokes]
        user_rng = np.random.default_rng(random_state + int(uid))
        shuffled = list(user_rng.permutation(unseen_jokes))
        random_ranked[uid] = shuffled

    _compute_metrics("Random Top-10", random_ranked)

    # 2) Content-based ranking using user train-history weights
    tfidf = TfidfVectorizer(stop_words="english", max_features=300)
    tfidf_matrix = tfidf.fit_transform(jokes_sorted["joke_text"].fillna(""))
    item_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    user_item_train = train_df.pivot_table(
        index="user_id",
        columns="joke_id",
        values="rating",
        aggfunc="mean",
    ).reindex(columns=joke_ids)

    content_ranked = {}
    for uid in eligible_eval_users:
        user_row = user_item_train.loc[uid]
        seen_jokes = train_seen_by_user[uid]
        seen_cols = [item_to_col[jid] for jid in seen_jokes if jid in item_to_col]
        if not seen_cols:
            content_ranked[uid] = _sorted_unseen_items(popularity_scores, seen_jokes)
            continue

        seen_ratings = np.array([float(user_row.iloc[col]) for col in seen_cols], dtype=float)
        shifted = (seen_ratings + 10.0) / 20.0
        if shifted.sum() <= 1e-12:
            weights = np.full(len(shifted), 1.0 / len(shifted), dtype=float)
        else:
            weights = shifted / shifted.sum()

        weighted_scores = np.average(item_sim[seen_cols], axis=0, weights=weights)
        score_map = {int(jid): float(weighted_scores[item_to_col[int(jid)]]) for jid in joke_ids}
        content_ranked[uid] = _sorted_unseen_items(score_map, seen_jokes)

    _compute_metrics("Content-Based", content_ranked)

    # 3) Collaborative filtering (user-based k-NN ranked retrieval)
    try:
        if verbose:
            print("Training collaborative model for evaluation...")

        user_item_cf = train_df.pivot_table(
            index="user_id",
            columns="joke_id",
            values="rating",
            aggfunc="mean",
        ).reindex(columns=joke_ids)

        joke_means_cf = user_item_cf.mean(axis=0)
        user_item_cf_filled = user_item_cf.fillna(joke_means_cf).fillna(global_mean)

        k_neighbors = min(40, len(user_item_cf_filled))
        nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
        nn_model.fit(user_item_cf_filled.values)

        eval_user_matrix = user_item_cf_filled.loc[eligible_eval_users].to_numpy(dtype=float)
        _, neighbor_indices = nn_model.kneighbors(eval_user_matrix, return_distance=True)
        user_item_values = user_item_cf.to_numpy(dtype=float)
        neighbor_ratings = user_item_values[neighbor_indices]
        mean_predictions = np.nanmean(neighbor_ratings, axis=1)
        joke_mean_values = joke_means_cf.to_numpy(dtype=float)
        mean_predictions = np.where(np.isnan(mean_predictions), joke_mean_values, mean_predictions)

        collaborative_ranked = {}
        for row_idx, uid in enumerate(eligible_eval_users):
            seen_jokes = train_seen_by_user[uid]
            score_row = mean_predictions[row_idx]
            score_map = {
                int(joke_id): float(score_row[col_idx])
                for col_idx, joke_id in enumerate(joke_ids)
                if int(joke_id) not in seen_jokes
            }

            collaborative_ranked[uid] = _sorted_unseen_items(score_map, seen_jokes)

        _compute_metrics("Collaborative k-NN", collaborative_ranked)
    except Exception as e:
        if verbose:
            print(f"Collaborative k-NN   ERROR: {e}")

    # 4) Deep learning (embedding-based ranked retrieval)
    if include_deep_learning:
        try:
            dl_result = deep_learning_recommender(train_df, jokes_df_eval)
            if dl_result is None:
                raise ValueError("Deep learning dependencies unavailable")

            item_embeddings = dl_result[1]
            emb_count = len(item_embeddings)
            item_norms = np.linalg.norm(item_embeddings, axis=1)

            user_embeddings = {}
            for uid, grp in train_df.groupby("user_id"):
                item_ids = grp["joke_id"].to_numpy(dtype=int)
                ratings_vals = grp["rating"].to_numpy(dtype=float)
                valid = (item_ids >= 0) & (item_ids < emb_count)
                if not np.any(valid):
                    continue

                vecs = item_embeddings[item_ids[valid]]
                w = ratings_vals[valid]
                if np.sum(np.abs(w)) < 1e-9:
                    user_embeddings[uid] = vecs.mean(axis=0)
                else:
                    # Signed weighting keeps dislike/like direction in user representation.
                    user_embeddings[uid] = np.average(vecs, axis=0, weights=w)

            deep_learning_ranked = {}
            for uid in eligible_eval_users:
                seen_jokes = train_seen_by_user[uid]
                if uid not in user_embeddings:
                    deep_learning_ranked[uid] = _sorted_unseen_items(popularity_scores, seen_jokes)
                    continue

                u = user_embeddings[uid]
                score_map = {}
                for joke_id in joke_ids:
                    joke_id_int = int(joke_id)
                    if joke_id_int in seen_jokes or joke_id_int >= emb_count:
                        continue

                    v = item_embeddings[joke_id_int]
                    denom = np.linalg.norm(u) * item_norms[joke_id_int]
                    if denom < 1e-12:
                        score = float(train_joke_means.get(joke_id_int, global_mean))
                    else:
                        score = float(np.dot(u, v) / denom)
                    score_map[joke_id_int] = score

                deep_learning_ranked[uid] = _sorted_unseen_items(score_map, seen_jokes)

            _compute_metrics("Deep Learning", deep_learning_ranked)
        except Exception as e:
            if verbose:
                print(f"Deep Learning        ERROR: {e}")

    if results:
        if verbose:
            print("\n" + "-" * 80)
            print(f"RANKING (higher Top@{top_n}, then Recall@{top_n}, is better)")
            print("-" * 80)
        ranked_results = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)
        for rank, (name, recall_score, top_score) in enumerate(ranked_results, start=1):
            if verbose:
                print(f"{rank}. {name:<20} Top@{top_n}={top_score:.4f} | Recall@{top_n}={recall_score:.4f}")

    if verbose:
        print("=" * 80)

    ranking = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)
    metrics = [
        {
            "method": name,
            "recall_score": round(recall_score, 4),
            "top_score": round(top_score, 4),
        }
        for name, recall_score, top_score in ranking
    ]

    return {
        "cache_version": EVALUATION_CACHE_VERSION,
        "metric_name": f"Recall@{top_n}",
        "top_metric_name": f"Top@{top_n}",
        "top_metric_definition": "For each user, rank only held-out test jokes by predicted score and compare predicted Top-10 with actual Top-10 by true test rating.",
        "top_n": int(top_n),
        "relevance_threshold": float(relevance_threshold),
        "split": {
            "train_count": int(len(train_df)),
            "test_count": int(len(test_df)),
            "train_percent": int((1 - test_size) * 100),
            "test_percent": int(test_size * 100),
            "random_state": int(random_state),
            "users_considered": int(users_considered),
            "users_skipped_for_split": int(users_skipped_for_split),
            "users_evaluated": int(len(eligible_eval_users)),
            "users_without_relevant_test": int(users_considered - len(eligible_eval_users)),
            "users_top_evaluated": int(len(top_eval_users)),
        },
        "metrics": metrics,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_or_build_evaluation_results(force_refresh=False):
    """Load cached evaluation results or compute them on demand."""
    cache_file = _cache_paths()["evaluation"]

    if cache_file.exists() and not force_refresh:
        with cache_file.open("rb") as f:
            cached = pickle.load(f)
        if _is_cached_evaluation_current(cached):
            return cached

    jokes_df_local, ratings_df_local = load_jester_dataset(use_cache=True)
    jokes_df_local, ratings_df_local = prepare_data_for_models(jokes_df_local, ratings_df_local)
    results = evaluate_recommenders(
        jokes_df_local,
        ratings_df_local,
        test_size=0.3,
        random_state=42,
        top_n=10,
        relevance_threshold=0.0,
        include_deep_learning=False,
        verbose=False,
    )

    with cache_file.open("wb") as f:
        pickle.dump(results, f)

    return results


def ensure_evaluation_results_cached():
    """Build the evaluation cache ahead of time so the UI only reads stored results."""
    cache_file = _cache_paths()["evaluation"]

    if cache_file.exists():
        with cache_file.open("rb") as f:
            cached = pickle.load(f)
        if _is_cached_evaluation_current(cached):
            print("Evaluation results cache is current.")
            return cached

    print("Precomputing evaluation results cache...")
    include_deep_learning = ARTIFACTS.get("item_embeddings") is not None if ARTIFACTS else False
    results = evaluate_recommenders(
        jokes_df,
        ratings_df,
        test_size=0.3,
        random_state=42,
        top_n=10,
        relevance_threshold=0.0,
        include_deep_learning=include_deep_learning,
        verbose=False,
    )

    with cache_file.open("wb") as f:
        pickle.dump(results, f)

    print("Evaluation results cached.")
    return results


def get_cached_evaluation_results():
    """Return cached evaluation results only. Do not compute on request."""
    cache_file = _cache_paths()["evaluation"]
    if not cache_file.exists():
        raise FileNotFoundError("Evaluation results cache is missing. Start the app once to precompute it.")

    with cache_file.open("rb") as f:
        cached = pickle.load(f)

    if not _is_cached_evaluation_current(cached):
        raise ValueError("Evaluation results cache is outdated. Restart the app to rebuild it.")

    return cached


@app.route('/evaluation-results', methods=['GET'])
def evaluation_results():
    """Return per-user joke-holdout Recall@10 results for all recommendation methods."""
    try:
        results = get_cached_evaluation_results()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import sys
    
    # Check if running as Flask app or standalone analysis
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Run standalone analysis
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        # Run objective 70/30 evaluation of all methods
        jokes_df_local, ratings_df_local = load_jester_dataset(use_cache=True)
        jokes_df_local, ratings_df_local = prepare_data_for_models(jokes_df_local, ratings_df_local)
        evaluate_recommenders(jokes_df_local, ratings_df_local, test_size=0.3, random_state=42, verbose=True)
    else:
        # Run Flask web app (default)
        init_app()
        app.run(host="0.0.0.0", debug=True, port=5000)
