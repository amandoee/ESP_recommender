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
    """Simple MLP Recommender model based on user-item concatenation"""
    print("--- MLP DEEP LEARNING MODEL ---")
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print(f"Error: tensorflow not available.")
        return None
    
    # 1. Data Preparation
    ratings_df_copy = ratings_df.dropna(subset=['rating']).copy()
    ratings_df_copy['user_id'] = ratings_df_copy['user_id'].astype(int)
    ratings_df_copy['joke_id'] = ratings_df_copy['joke_id'].astype(int)

    num_users = ratings_df_copy['user_id'].max() + 1
    num_items = ratings_df_copy['joke_id'].max() + 1

    # Hyperparameters
    embedding_dim = 32 # Dimension for p_u and q_i

    # 2. Embedding Layer (Step 1 in your description)
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_emb = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_emb')(user_input)
    item_emb = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_emb')(item_input)

    user_flat = Flatten()(user_emb)
    item_flat = Flatten()(item_emb)

    # 3. Concatenation + Normalization
    mlp_vector = Concatenate()([user_flat, item_flat])
    mlp_vector = BatchNormalization()(mlp_vector)

    # 4. Dense Layers with Dropout (The "Anti-Overfit" layers)
    mlp_vector = Dense(64, activation='relu')(mlp_vector)
    mlp_vector = Dropout(0.4)(mlp_vector) # Randomly "shuts off" 40% of neurons
    
    mlp_vector = Dense(32, activation='relu')(mlp_vector)
    mlp_vector = Dropout(0.2)(mlp_vector)

    # 5. Prediction Layer (Sigmoid for [0, 1] range)
    prediction = Dense(1, activation='sigmoid', name='prediction')(mlp_vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 6. Early Stopping: Stops training once val_loss stops improving
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )

    # Use the actual column values from the DataFrame, not the Keras Input objects
    X_user = ratings_df_copy['user_id'].values
    X_item = ratings_df_copy['joke_id'].values
    y_train_scaled = (ratings_df_copy['rating'].values + 10.0) / 20.0

    # Correct model.fit call
    model.fit(
        [X_user, X_item], # These are your inputs
        y_train_scaled,   # These are your targets
        batch_size=2048,
        epochs=20,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
        
    return model


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
    print("Popularity ranking")

    # 2) Content-based matrices (fit once).
    tfidf = TfidfVectorizer(stop_words="english", max_features=300)
    texts = jokes["joke_text"].fillna("")
    tfidf_matrix = tfidf.fit_transform(texts)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print("Content-based matrices")

    # 3) User-based collaborative artifacts (train NearestNeighbors once).
    user_counts = ratings.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 5].index.to_numpy()
    sampled = ratings[ratings["user_id"].isin(eligible_users)].copy()
    print("CF artifacts")

    print(f"Training user-based k-NN model on all {len(eligible_users)} eligible users...")
    
    # Build user-item matrix for similarity-based predictions
    user_item = sampled.pivot_table(index="user_id", columns="joke_id", values="rating", aggfunc="mean")
    user_item = user_item.reindex(columns=range(100))
    joke_means = user_item.mean(axis=0)
    user_item_filled = user_item.fillna(joke_means).fillna(0.0)

    k_neighbors = min(40, len(user_item_filled))
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn_model.fit(user_item_filled.values)

    # 4) Deep learning recommender
    dl_model = deep_learning_recommender(ratings, jokes)
    return {
        "popularity_table": popularity_table,
        "candidate_joke_ids": set(popularity_table["joke_id"].astype(int).tolist()),
        "tfidf": tfidf,
        "cosine_sim": cosine_sim,
        "user_item": user_item,
        "user_item_filled": user_item_filled,
        "joke_means": joke_means,
        "nn_model": nn_model,
        "dl_model": dl_model, # Changed from item_embeddings
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

@app.route('/evaluation-results')
def get_evaluation():
    try:
        cache_file = _cache_paths()["evaluation"]
        with cache_file.open("rb") as f:
            cached = pickle.load(f)

        meta = cached["split_meta"]
        response_data = {
            "metrics": [
                {"method": name, "recall_score": float(r), "top_score": float(t)}
                for name, r, t in cached["results"]
            ],
            "split": {
                "train_percent": 70,
                "test_percent": 30,
                "train_count": meta["train_count"],
                "test_count": meta["test_count"],
                "users_evaluated": meta["users_evaluated"],
                "users_top_evaluated": meta["users_top_evaluated"],
                "random_state": 42
            },
            "metric_name": "Recall@10",
            "top_metric_name": "Top@10",
            "top_metric_definition": "Search space restricted to user's test set items only",
            "relevance_threshold": 0.0,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Evaluation Error: {e}")
        return jsonify({"error": str(e)}), 500

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
            dl_model = ARTIFACTS.get("dl_model")
            if dl_model is None:
                raise ValueError("Deep learning model unavailable (TensorFlow not installed or training failed)")

            # Cold-start workaround for NeuMF: 
            # Borrow the most similar user ID found during the CF step to act as a proxy
            try:
                proxy_user_id = int(neighbor_ids[0])
            except (NameError, IndexError):
                proxy_user_id = 0 # Fallback if CF failed

            unrated_jokes = list(candidate_jokes - rated_jokes)
            
            # Prepare batch arrays for the model
            user_array = np.array([proxy_user_id] * len(unrated_jokes))
            item_array = np.array(unrated_jokes)

            # Predict ratings in one batch
            preds = dl_model.predict([user_array, item_array], verbose=0).flatten()
            
            predictions_dl = [
                {'joke_id': int(jid), 'pred_rating': float(pred)} 
                for jid, pred in zip(unrated_jokes, preds)
            ]

            max_score = max((p['pred_rating'] for p in predictions_dl), default=1.0)
            top_3_dl = sorted(predictions_dl, key=lambda x: x['pred_rating'], reverse=True)[:3]
            
            recommendations['deep_learning'] = [
                {
                    'joke_id': rec['joke_id'],
                    'joke_text': jokes_df[jokes_df['joke_id'] == rec['joke_id']]['joke_text'].values[0],
                    'confidence': round(max(0, rec['pred_rating']) / max(1e-9, abs(max_score)) * 100, 1),
                    'reasoning': f"Based on deep non-linear pattern matching predicting what score"
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
def evaluate_recommenders(jokes_df_eval, train_df, test_df, top_n=10, relevance_threshold=0.0, include_deep_learning=True, verbose=True):
    """
    Complete evaluation suite: Popularity, Content-Based, Collaborative (k-NN), and MLP.
    All models are trained ONLY on train_df and evaluated against test_df.
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"OBJECTIVE EVALUATION (RECALL@{top_n} & TOP@{top_n})")
        print("=" * 80)

    # 1. Setup metadata
    joke_ids = np.sort(train_df["joke_id"].unique())
    item_to_col = {int(jid): idx for idx, jid in enumerate(joke_ids)}
    global_mean = train_df["rating"].mean()
    train_joke_means = train_df.groupby("joke_id")["rating"].mean()
    
    # Track which jokes each user has already seen in training
    train_seen_by_user = train_df.groupby("user_id")["joke_id"].apply(set).to_dict()
    
    # Identify users eligible for Recall evaluation (those with at least one 'relevant' test joke)
    test_by_user = {uid: grp for uid, grp in test_df.groupby("user_id")}
    eligible_eval_users = [
        uid for uid, grp in test_by_user.items() 
        if (grp["rating"] > relevance_threshold).any()
    ]

    results = []

    def _compute_metrics(name, user_rankings):
        """
        user_rankings: dict where keys are user_ids and values are 
        all unseen joke_ids sorted by predicted score (descending).
        """
        recalls = []
        top_hits = []

        # Users for Recall: Must have at least one relevant test item (> 0.0)
        users_with_relevant = [u for u in eligible_eval_users if u in user_rankings]
        
        # Users for Top@N: All users with at least one item in the test set
        users_with_test = [u for u in test_by_user.keys() if u in user_rankings]

        # 1. Compute Recall@N
        for uid in users_with_relevant:
            # TopN(u) excluding items seen during training
            top_n_rec = user_rankings[uid][:top_n]
            
            # Relevant(u): items in test set > threshold
            relevant_items = set(test_by_user[uid].loc[test_by_user[uid]["rating"] > relevance_threshold, "joke_id"])
            
            hit_count = len(set(top_n_rec) & relevant_items)
            recalls.append(hit_count / len(relevant_items))

        # 2. Compute Top@N
        for uid in users_with_test:
            user_test_df = test_by_user[uid]
            test_joke_ids = set(user_test_df["joke_id"].unique())
            
            # ActualTop(u): Top items from the user's test set sorted by rating
            actual_top = set(user_test_df.sort_values(by="rating", ascending=False)["joke_id"].iloc[:top_n])
            
            # TopN(u): Filter predictions to ONLY include items in the user's test set
            # Then take the top N of those
            predicted_in_test = [jid for jid in user_rankings[uid] if jid in test_joke_ids]
            predicted_top_test = set(predicted_in_test[:top_n])
            
            # Intersection
            top_hit_rate = len(predicted_top_test & actual_top) / len(actual_top)
            top_hits.append(top_hit_rate)

        recall_final = np.mean(recalls) if recalls else 0
        top_final = np.mean(top_hits) if top_hits else 0
        
        results.append((name, recall_final, top_final))
        print(f"{name:<20} Recall@{top_n}={recall_final:.4f} | Top@{top_n}={top_final:.4f}")

    # --- METHOD 1: Popularity (Baseline) ---
    pop_scores = {jid: train_joke_means.get(jid, global_mean) for jid in joke_ids}
    pop_ranked = {}
    for uid in eligible_eval_users:
        seen = train_seen_by_user.get(uid, set())
        # Sort all jokes by pop, then filter seen
        sorted_jokes = sorted(pop_scores, key=pop_scores.get, reverse=True)
        pop_ranked[uid] = [j for j in sorted_jokes if j not in seen]
    _compute_metrics("Popularity", pop_ranked)

    # --- METHOD 2: Content-Based ---
    tfidf = TfidfVectorizer(stop_words="english", max_features=300)
    # Sort jokes to match the joke_ids order for the similarity matrix
    jokes_sorted = jokes_df_eval[jokes_df_eval["joke_id"].isin(joke_ids)].sort_values("joke_id")
    tfidf_matrix = tfidf.fit_transform(jokes_sorted["joke_text"].fillna(""))
    item_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    content_ranked = {}
    for uid in eligible_eval_users:
        user_ratings = train_df[train_df['user_id'] == uid]
        seen = train_seen_by_user.get(uid, set())
        
        if user_ratings.empty:
            content_ranked[uid] = pop_ranked.get(uid, [])
            continue

        # 1. Shift ratings from [-10, 10] to [0, 20] 
        # 2. Add a tiny value (1e-9) to avoid the ZeroDivisionError
        shifted_ratings = user_ratings['rating'].values + 10.0
        weights = shifted_ratings + 1e-9 
        
        indices = [item_to_col[jid] for jid in user_ratings['joke_id'] if jid in item_to_col]
        
        if not indices:
            content_ranked[uid] = pop_ranked.get(uid, [])
            continue
            
        # This will now safely normalize because weights.sum() > 0
        user_profile = np.average(item_sim[indices], axis=0, weights=weights)
        
        score_map = {jid: user_profile[item_to_col[jid]] for jid in joke_ids if jid not in seen}
        content_ranked[uid] = sorted(score_map, key=score_map.get, reverse=True)
    _compute_metrics("Content-Based", content_ranked)

    # --- METHOD 3: Collaborative Filtering (k-NN) ---
    user_item_matrix = train_df.pivot(index="user_id", columns="joke_id", values="rating")
    # Fill with item means for the distance calculation
    user_item_filled = user_item_matrix.fillna(train_joke_means).fillna(global_mean)
    
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=40)
    nn.fit(user_item_filled.values)

    cf_ranked = {}
    # Get neighbor indices for all eligible users at once for speed
    user_indices = [user_item_filled.index.get_loc(uid) for uid in eligible_eval_users]
    distances, indices = nn.kneighbors(user_item_filled.iloc[user_indices])

    for i, uid in enumerate(eligible_eval_users):
        seen = train_seen_by_user.get(uid, set())
        # Average ratings from the 40 most similar users
        neighbor_ratings = user_item_matrix.iloc[indices[i]].mean(axis=0)
        # Fallback to joke means for jokes the neighbors haven't rated
        pred_scores = neighbor_ratings.fillna(train_joke_means).fillna(global_mean)
        
        score_map = {jid: pred_scores.get(jid, global_mean) for jid in joke_ids if jid not in seen}
        cf_ranked[uid] = sorted(score_map, key=score_map.get, reverse=True)
    _compute_metrics("Collaborative k-NN", cf_ranked)

    # --- METHOD 4: MLP Deep Learning ---
    if include_deep_learning:
        dl_model = ARTIFACTS.get("dl_model")
        if dl_model:
            print("Generating MLP predictions (Vectorized)...")
            dl_ranked = {}
            all_u, all_j = [], []
            for uid in eligible_eval_users:
                for jid in joke_ids:
                    all_u.append(uid); all_j.append(jid)
            
            preds = dl_model.predict([np.array(all_u), np.array(all_j)], batch_size=4096, verbose=0).flatten()
            
            idx = 0
            for uid in eligible_eval_users:
                seen = train_seen_by_user.get(uid, set())
                user_scores = []
                for jid in joke_ids:
                    if jid not in seen:
                        user_scores.append((jid, preds[idx]))
                    idx += 1
                user_scores.sort(key=lambda x: x[1], reverse=True)
                dl_ranked[uid] = [x[0] for x in user_scores]
            
            _compute_metrics("MLP Deep Learning", dl_ranked)

    return results

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
        include_deep_learning=True,
        verbose=False,
    )

    with cache_file.open("wb") as f:
        pickle.dump(results, f)

    return results


def ensure_evaluation_results_cached(force_refresh=False):
    cache_file = _cache_paths()["evaluation"]

    if cache_file.exists() and not force_refresh:
        with cache_file.open("rb") as f:
            cached = pickle.load(f)
        if _is_cached_evaluation_current(cached):
            print("Evaluation results cache is current.")
            return cached["results"]   # ← return just the list

    print("Refreshing evaluation results cache...")
    dl_model = ARTIFACTS.get("dl_model")
    train_df, test_df = split_data_per_user(ratings_df, test_size=0.3)

    results = evaluate_recommenders(
        jokes_df_eval=jokes_df,
        train_df=train_df,
        test_df=test_df,
        top_n=10,
        relevance_threshold=0.0,
        include_deep_learning=(dl_model is not None),
        verbose=True,
    )

    # Save as a dict so _is_cached_evaluation_current() can validate it
    train_df, test_df = split_data_per_user(ratings_df, test_size=0.3)
    test_by_user = {uid: grp for uid, grp in test_df.groupby("user_id")}
    users_eval = [uid for uid, grp in test_by_user.items() if (grp["rating"] > 0.0).any()]

    payload = {
        "cache_version": EVALUATION_CACHE_VERSION,
        "metric_name": "Recall@10",
        "top_metric_name": "Top@10",
        "results": results,
        # Save counts so the route never has to re-split
        "split_meta": {
            "train_count": len(train_df),
            "test_count": len(test_df),
            "users_evaluated": len(users_eval),
            "users_top_evaluated": len(test_by_user),
        }
    }
    with cache_file.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Evaluation results cached to {cache_file}")
    return results   # ← always return just the list


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

"""
@app.route('/evaluation-results', methods=['GET'])
def evaluation_results():
    try:
        results = get_cached_evaluation_results()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""
def main():
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

def split_data_per_user(ratings_df, test_size=0.3, random_state=42):
    """Splits ratings into train/test, ensuring each user has items in both."""
    rng = np.random.default_rng(random_state)
    train_list, test_list = [], []

    for _, group in ratings_df.groupby("user_id"):
        if len(group) < 2:
            train_list.append(group) # Can't split 1 rating
            continue
            
        # Shuffle and split
        shuffled = group.sample(frac=1, random_state=random_state)
        n_test = max(1, int(len(group) * test_size))
        
        test_list.append(shuffled.iloc[:n_test])
        train_list.append(shuffled.iloc[n_test:])

    return pd.concat(train_list), pd.concat(test_list)


if __name__ == '__main__':
    import sys
    
    # Check if running as Flask app or standalone analysis
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Run standalone analysis
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        print("evaluating models")
        # Run objective 70/30 evaluation of all methods
        jokes_df_local, ratings_df_local = load_jester_dataset(use_cache=True)
        jokes_df_local, ratings_df_local = prepare_data_for_models(jokes_df_local, ratings_df_local)


        # 1. SPLIT DATA FIRST (To prevent leakage)
        train_df, test_df = split_data_per_user(ratings_df_local, test_size=0.3)
        
        # 2. TRAIN MLP ON TRAIN_DF ONLY
        # This prevents the model from "memorizing" the test answers
        dl_model = deep_learning_recommender(train_df, jokes_df_local)
        ARTIFACTS["dl_model"] = dl_model

        #evaluate_recommenders(jokes_df_local, ratings_df_local, test_size=0.3, random_state=42, verbose=True)
        evaluate_recommenders(
            jokes_df_eval=jokes_df_local, 
            train_df=train_df, 
            test_df=test_df, 
            verbose=True
        )
    else:
        # Run Flask web app (default)
        init_app()
        app.run(host="0.0.0.0", debug=True, port=5000)
