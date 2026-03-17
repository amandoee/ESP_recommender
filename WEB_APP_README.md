# 😂 Jester Joke Recommender System

An interactive web application showcasing 4 different joke recommendation algorithms using the Jester dataset.

## Features

This project demonstrates and compares 4 distinct recommendation approaches:

### 1. **Popularity-Based Recommendations** 📊
- Recommends jokes with the highest average ratings across all users
- Simple, interpretable, and works well for new users
- **Pros:** Fast, reliable baseline
- **Cons:** No personalization

### 2. **Content-Based Filtering** 📝
- Analyzes joke text using TF-IDF and recommends jokes with similar keywords/themes
- Uses cosine similarity to match textual content
- **Pros:** Personalized, explains why recommendations are made
- **Cons:** Requires good joke descriptions

### 3. **User-Based Collaborative Filtering** 👥
- Finds users with similar rating patterns and recommends jokes they liked
- Uses k-NN with cosine similarity (k=40, min_support=3)
- Trained on sampled users (5000) to manage memory usage
- **Pros:** Finds similar tastes, very effective
- **Cons:** Requires many user ratings, cold-start problem

### 4. **Deep Learning Model** 🤖
- Two-tower neural network that learns latent factors from user-item interactions
- Uses TensorFlow Recommenders embeddings
- **Pros:** Learns complex patterns, handles sparsity
- **Cons:** Computationally expensive, black-box

## Getting Started

### Prerequisites
- Python 3.12
- UV package manager

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Ensure you have the dataset files:
   - `Dataset4JokeSet/Dataset4JokeSet.xlsx` (158 jokes)
   - `jester_dataset_1_1/jester-data-1.xls` (24,983 users)
   - `jester_dataset_1_2/jester-data-2.xls` (23,500 users)
   - `jester_dataset_1_3/jester-data-3.xls` (24,938 users)

### Running the Web Application

```bash
uv run python app.py
```

Then open your browser and navigate to: **http://localhost:5000**

### Usage

1. **Rate Jokes**: The app displays 10 random jokes. Rate each one using the emoji scale:
   - 😢 = -10 (strongly dislike)
   - 😕 = -5 (dislike)
   - 😐 = 0 (neutral)
   - 😊 = 5 (like)
   - 😍 = 10 (strongly like)

2. **Get Recommendations**: Click "Get Recommendations" to see personalized suggestions from all 4 methods

3. **Compare Methods**: View how different algorithms recommend different jokes based on your ratings

## Running the Standalone Script

To run the analysis without the web interface:

```bash
uv run python main.py
```

This will:
- Load the full Jester dataset (73,421 users, 4.1M ratings)
- Display top 5 most popular jokes
- Show 5 jokes similar to joke #0
- Train user-based k-NN on sampled data
- Output accuracy metrics (RMSE, MAE)

## Dataset Information

**Jester Dataset (Version 1)**
- **Jokes**: 158 unique jokes with text content
- **Users**: 73,421 unique users across 3 datasets
- **Ratings**: ~7.3M total ratings (4.1M valid after cleaning)
- **Scale**: -10 to +10 (99 = not rated)
- **Sparsity**: ~56% of user-joke combinations are unrated

### Dataset Breakdown
- **Dataset 1**: 24,983 users who rated ≥36 jokes
- **Dataset 2**: 23,500 users who rated ≥36 jokes
- **Dataset 3**: 24,938 users who rated 15-35 jokes

## Performance Metrics

### User-Based Collaborative Filtering (k-NN on 5000 sampled users)
- **RMSE**: ~2.5 (on -10 to +10 scale)
- **MAE**: ~2.1
- **Similarity**: Cosine with min_support=3
- **Training Time**: ~30-60 seconds

## Project Structure

```
Recommender/
├── main.py                 # Core recommendation algorithms
├── app.py                  # Flask web application
├── pyproject.toml          # Dependencies and project config
├── templates/
│   └── index.html          # Frontend interface
├── Dataset4JokeSet/
│   └── Dataset4JokeSet.xlsx       # Joke texts
├── jester_dataset_1_1/
│   └── jester-data-1.xls          # User ratings
├── jester_dataset_1_2/
│   └── jester-data-2.xls          # User ratings
└── jester_dataset_1_3/
    └── jester-data-3.xls          # User ratings
```

## Algorithm Details

### Data Loading
- Loads Excel files (.xlsx and .xls formats)
- Converts from matrix format (users × jokes) to long format (user_id, joke_id, rating)
- Cleans missing values (99 placeholders → NaN, then removed)
- Results: 73,421 users × 100 jokes matrix flattened to 4.1M valid ratings

### Preprocessing
- Removes ratings with missing values (3.2M entries)
- Filters users with <5 ratings to avoid sparsity in k-NN
- Samples 5000 users from remaining ~45K for memory efficiency

### Memory Management
- Full user-similarity matrix would be 73.4K × 73.4K = 40.2 GB
- Solution: Sample 5000 users → 5K × 5K matrix = ~200 MB
- Item-similarity matrix: 100 × 100 → Only 80 KB (alternative approach)

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: TF-IDF vectorization and cosine similarity
- `surprise`: Collaborative filtering algorithms
- `tensorflow`: Deep learning models
- `tensorflow-recommenders`: Neural recommendation models
- `flask`: Web framework
- `openpyxl`: Excel file reading
- `xlrd`: Legacy Excel file support

## Notes

- The web app trains the k-NN model on startup (takes ~30-60 seconds)
- Deep learning model is included but may have compatibility issues with Keras 3
- Each session shows 10 random jokes; ratings are not persisted
- Recommendations are computed on-the-fly from your ratings

## Runtime Optimization

To minimize runtime work, the app now precomputes and caches artifacts in `.cache/`:

- `jokes_df.pkl` and `ratings_df.pkl`: Parsed dataset cache (avoids repeated Excel parsing)
- `precomputed_artifacts.pkl`: Popularity ranking, content similarity matrix, collaborative neighbor index, and latent-factor model

At request time, the API only does lightweight operations:

- Filter pre-sorted popularity table
- Average a few precomputed similarity rows
- Query pre-fit nearest-neighbor index
- Score against precomputed latent item factors

This keeps `/get-recommendations` fast and avoids retraining/reclustering per request.

## References

- **Jester Dataset**: https://goldberg.berkeley.edu/jester-data/
- **Collaborative Filtering**: Surprise library documentation
- **TF-IDF**: Scikit-learn text feature extraction
- **Neural Recommendations**: TensorFlow Recommenders

## Author

Created for ESL - Aarhus University

## License

Educational use only
