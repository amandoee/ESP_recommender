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

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

warnings.filterwarnings("ignore")

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

def memory_based_cf(ratings_df, num_users=5000, min_ratings_per_user=5):
    """User-based collaborative filtering using your loaded ratings data
    
    Args:
        ratings_df: DataFrame with user, joke, and rating columns
        num_users: Number of users to sample (default 5000 to keep similarity matrix manageable)
        min_ratings_per_user: Minimum ratings per user to avoid sparsity issues (default 5)
    """
    print("--- MEMORY-BASED COLLABORATIVE FILTERING (User-Based k-NN) ---")

    # Lazy import keeps web app startup lightweight.
    from surprise import Dataset, Reader, KNNWithMeans, accuracy
    from surprise.model_selection import train_test_split
    
    # Filter users with at least min_ratings_per_user ratings to avoid sparsity
    user_counts = ratings_df.groupby('user_id').size()
    users_with_enough_ratings = user_counts[user_counts >= min_ratings_per_user].index
    ratings_filtered = ratings_df[ratings_df['user_id'].isin(users_with_enough_ratings)].copy()
    
    print(f"Filtered to users with ≥{min_ratings_per_user} ratings: {len(users_with_enough_ratings)} users")
    
    # Sample users to avoid memory issues
    sampled_users = np.random.choice(
        users_with_enough_ratings, 
        size=min(num_users, len(users_with_enough_ratings)), 
        replace=False
    )
    ratings_sample = ratings_filtered[ratings_filtered['user_id'].isin(sampled_users)].copy()
    
    print(f"Sampled {len(sampled_users)} users")
    print(f"Using {len(ratings_sample)} ratings for training")
    
    # Create a Surprise dataset from sampled ratings
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(
        ratings_sample[['user_id', 'joke_id', 'rating']], 
        reader
    )
    
    # Split data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    # User-based similarity with min_common_neighbors to handle sparsity
    sim_options = {
        'name': 'cosine', 
        'user_based': True,
        'min_support': 3  # Require at least 3 common ratings between users
    }
    algo = KNNWithMeans(sim_options=sim_options, k=40, verbose=False)
    
    # Train and test
    print("Training user-based k-NN model on sampled data...")
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Calculate Accuracy using RMSE and MAE
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    print(f"ACCURACY METRICS: RMSE = {rmse:.4f} | MAE = {mae:.4f}")
    print(f"EXPLAINABILITY: We recommend jokes based on users with similar rating patterns to you.\n")
    
    return algo


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
        "knn_model": base / "knn_model.pkl",
    }


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

    # 3) User-based collaborative artifacts (train KNNWithMeans once).
    from surprise import Dataset, Reader, KNNWithMeans
    
    user_counts = ratings.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 5].index.to_numpy()
    rng = np.random.default_rng(42)
    sample_n = min(4000, len(eligible_users))
    sampled_users = rng.choice(eligible_users, size=sample_n, replace=False)
    sampled = ratings[ratings["user_id"].isin(sampled_users)].copy()

    print(f"Training user-based k-NN model on {len(sampled_users)} sampled users...")
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(sampled[['user_id', 'joke_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    sim_options = {
        'name': 'cosine', 
        'user_based': True,
        'min_support': 3
    }
    knn_model = KNNWithMeans(sim_options=sim_options, k=40, verbose=False)
    knn_model.fit(trainset)
    
    # Build user-item matrix for similarity-based predictions
    user_item = sampled.pivot_table(index="user_id", columns="joke_id", values="rating", aggfunc="mean")
    user_item = user_item.reindex(columns=range(100))
    joke_means = user_item.mean(axis=0)
    user_item_filled = user_item.fillna(joke_means).fillna(0.0)

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
        "knn_model": knn_model,
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
    knn_cache_file = cache_paths["knn_model"]
    
    if cache_file.exists() and knn_cache_file.exists():
        print("Loading precomputed artifacts from cache...")
        with cache_file.open("rb") as f:
            ARTIFACTS = pickle.load(f)
        with knn_cache_file.open("rb") as f:
            ARTIFACTS["knn_model"] = pickle.load(f)
        required_keys = {"popularity_table", "candidate_joke_ids", "cosine_sim", "user_item", "user_item_filled", "joke_means", "knn_model", "item_embeddings"}
        if not required_keys.issubset(set(ARTIFACTS.keys())):
            print("Cached artifacts are outdated. Rebuilding...")
            ARTIFACTS = _build_artifacts(jokes_df, ratings_df)
            with cache_file.open("wb") as f:
                pickle.dump({k: v for k, v in ARTIFACTS.items() if k != "knn_model"}, f)
            with knn_cache_file.open("wb") as f:
                pickle.dump(ARTIFACTS["knn_model"], f)
    else:
        ARTIFACTS = _build_artifacts(jokes_df, ratings_df)
        with cache_file.open("wb") as f:
            pickle.dump({k: v for k, v in ARTIFACTS.items() if k != "knn_model"}, f)
        with knn_cache_file.open("wb") as f:
            pickle.dump(ARTIFACTS["knn_model"], f)

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
        
        # 3. COLLABORATIVE FILTERING RECOMMENDATIONS (using surprise library's KNNWithMeans)
        try:
            user_item = ARTIFACTS["user_item"]
            user_item_filled = ARTIFACTS["user_item_filled"]
            knn_model = ARTIFACTS["knn_model"]
            joke_means = ARTIFACTS["joke_means"]

            # For new users, compute similarity to training set users using cosine similarity
            filled_user = user_vector.copy()
            for jid in range(100):
                if pd.isna(filled_user.loc[jid]):
                    filled_user.loc[jid] = joke_means.loc[jid] if not pd.isna(joke_means.loc[jid]) else 0.0

            # Compute cosine similarities between new user and training set users
            similarities = cosine_similarity([filled_user.values], user_item_filled.values)[0]
            
            # Get k=40 nearest neighbors (same as KNNWithMeans k parameter)
            k = min(40, len(similarities))
            nearest_indices = np.argsort(similarities)[-k:][::-1]
            neighbor_ids = user_item_filled.index[nearest_indices]

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


if __name__ == '__main__':
    import sys
    
    # Check if running as Flask app or standalone analysis
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Run standalone analysis
        main()
    else:
        # Run Flask web app (default)
        init_app()
        app.run(host="0.0.0.0", debug=True, port=5000)
