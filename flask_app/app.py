from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import os

# Load vectorizer and model at startup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

with open(os.path.join(ROOT_DIR, 'data/processed/bow_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

with open(os.path.join(ROOT_DIR, 'models/lgbm_Youtube_sentiment.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(ROOT_DIR, 'data/processed/X_train_BOW_custom.pkl'), 'rb') as f:
    X_train_features_ref = pickle.load(f)

# Number of BOW and custom features
n_bow_features = vectorizer.transform(['dummy']).shape[1] #Transforms a dummy string to get number of BOW features (shape[1] = number of columns).
n_total_features = X_train_features_ref.shape[1]
n_custom_features = n_total_features - n_bow_features

app = Flask(__name__)

def extract_custom_features(texts):
    """Replace with your actual extractor if available."""
    # Example: zero-padding if extractor unavailable
    return pd.DataFrame(
        np.zeros((len(texts), n_custom_features), dtype=np.float32),
        columns=[f"custom_feature_{i}" for i in range(n_custom_features)]
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments', [])

    # BOW features
    X_bow = vectorizer.transform(comments)

    # Custom features
    custom_df = extract_custom_features(comments)
    custom_np = custom_df.fillna(0).astype(np.float32).values

    # Pad/truncate custom features if needed
    if custom_np.shape[1] != n_custom_features:
        if custom_np.shape[1] > n_custom_features:
            custom_np = custom_np[:, :n_custom_features]
        else:
            pad_width = n_custom_features - custom_np.shape[1]
            custom_np = np.hstack([custom_np, np.zeros((custom_np.shape[0], pad_width), dtype=np.float32)])

    X_custom = csr_matrix(custom_np)

    # Combine BOW + custom
    X_final = hstack([X_bow, X_custom])

    # Use the same column names as training
    bow_feature_names = vectorizer.get_feature_names_out()
    custom_feature_names = [f"custom_feature_{i}" for i in range(n_custom_features)]
    all_feature_names = list(bow_feature_names) + custom_feature_names

    X_final_df = pd.DataFrame(X_final.toarray(), columns=all_feature_names)

    # Predict
    preds = model.predict(X_final_df).tolist()
    # Build response
    response = [{"comment": comment, "sentiment": int(preds[i])} for i, comment in enumerate(comments)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)







