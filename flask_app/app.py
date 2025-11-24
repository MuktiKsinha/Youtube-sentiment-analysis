import os
import pickle
import io
import logging

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import spacy
from nltk.stem import WordNetLemmatizer

import mlflow

# ----------------------- Logging & Config -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

VECTORIZER_PATH = os.path.join(ROOT_DIR, 'data/processed/bow_vectorizer.pkl')
X_TRAIN_REF_PATH = os.path.join(ROOT_DIR, 'data/processed/X_train_BOW_custom.pkl')

MLFLOW_TRACKING_URI = "http://ec2-13-60-52-178.eu-north-1.compute.amazonaws.com:5000/"
MODEL_NAME = "Youtube_chrome_plugin_model"
MODEL_VERSION = "4"

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# ----------------------- Load Model & Vectorizer -----------------------
def load_model_and_vectorizer():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    logger.info(f"Loading model from MLflow: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("MLflow model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load MLflow model.")
        raise RuntimeError(f"Cannot load model from MLflow registry: {e}")

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info("Loaded vectorizer")

    # Load training reference DataFrame with column names
    with open(X_TRAIN_REF_PATH, 'rb') as f:
        X_train_features_ref = pickle.load(f)
    if not isinstance(X_train_features_ref, pd.DataFrame):
        raise ValueError("X_train_features_ref must be a pandas DataFrame with column names.")
    logger.info("Loaded training feature reference matrix")

    bow_columns = list(vectorizer.get_feature_names_out())
    custom_columns = [c for c in X_train_features_ref.columns if c not in bow_columns]

    return model, vectorizer, X_train_features_ref, bow_columns, custom_columns

model, vectorizer, X_train_features_ref, bow_columns, custom_columns = load_model_and_vectorizer()

# ----------------------- Custom Feature Extraction -----------------------
def extract_custom_features(texts):
    results = []
    for doc in nlp.pipe(texts, batch_size=32):
        word_list = [token.text for token in doc]
        word_count = len(word_list)
        unique_words = len(set(word_list))
        pos_tags = [token.pos_ for token in doc]

        features = {
            "comment_length": len(doc.text),
            "word_count": word_count,
            "avg_word_length": sum(len(w) for w in word_list)/word_count if word_count else 0,
            "unique_word_count": unique_words,
            "lexical_diversity": unique_words/word_count if word_count else 0,
            "pos_count": len(pos_tags)
        }

        for tag in set(pos_tags):
            features[f"pos_ratio_{tag}"] = pos_tags.count(tag)/word_count

        results.append(features)

    df = pd.DataFrame(results).fillna(0)
    df = df.reindex(columns=custom_columns, fill_value=0)  # match training columns
    return df

# ----------------------------- Build Full Feature DataFrame -----------------------------
def build_feature_dataframe(comments):
    cleaned_comments = [c for c in comments if c.strip()]
    if not cleaned_comments:
        return None, []

    # BOW DataFrame
    X_bow_sparse = vectorizer.transform(cleaned_comments)
    X_bow_df = pd.DataFrame(X_bow_sparse.toarray(), columns=bow_columns)

    # Custom features
    custom_df = extract_custom_features(cleaned_comments)

    # Combine BOW + custom features
    X_full = pd.concat([X_bow_df, custom_df], axis=1)

    # Ensure all columns of training set are present
    X_full = X_full.reindex(columns=X_train_features_ref.columns, fill_value=0)

    return X_full, cleaned_comments

# ----------------------- Flask App -----------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/")
def home():
    return "Flask Sentiment API (BOW + spaCy Custom Features)"

# ----------------------------- /predict_with_timestamps -----------------------------
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        data = request.get_json()
        items = data.get("comments", [])

        comments = [it.get("text", "") for it in items]
        timestamps = [it.get("timestamp") for it in items]

        X_full, valid_comments = build_feature_dataframe(comments)
        if X_full is None:
            return jsonify([])

        preds = [int(p) for p in model.predict(X_full)]

        result = []
        valid_idx = 0
        for i, c in enumerate(comments):
            if c.strip():
                result.append({
                    "comment": c,
                    "sentiment": preds[valid_idx],
                    "timestamp": timestamps[i]
                })
                valid_idx += 1
            else:
                result.append({
                    "comment": c,
                    "sentiment": None,
                    "timestamp": timestamps[i]
                })

        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction with timestamps failed")
        return jsonify({"error": str(e)}), 500

# ----------------------------- /predict -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        comments = data.get("comments", [])

        X_full, valid_comments = build_feature_dataframe(comments)
        if X_full is None:
            return jsonify([])

        preds = [int(p) for p in model.predict(X_full)]

        result = []
        valid_idx = 0
        for c in comments:
            if c.strip():
                result.append({"comment": c, "sentiment": preds[valid_idx]})
                valid_idx += 1
            else:
                result.append({"comment": c, "sentiment": None})

        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

# ----------------------- Run Server -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)







