
</details>

---

## 🧪 Experiments & Model Performance

Multiple experimentation cycles improved model accuracy from **69% → 87%**.  
All experiments are tracked via **MLflow**.

| Version | Model | Vectorizer | Accuracy | Notes |
|--------:|-------|------------|---------:|------|
| v1 | Random Forest | CountVectorizer | 69% | Baseline |
| v2 | Logistic Regression | BOW | 77% | Text cleaning & lemmatization |
| v3 | Random Forest | BOW | 82% | Class balancing, hyperparam tuning |
| v4 | LightGBM | BOW + n-grams | 87% | Emoji normalization & sarcasm handling |

**Final Model Metrics (Macro Avg)**:  
- **Precision:** 0.86  
- **Recall:** 0.85  
- **F1-score:** 0.85

---

## 🚀 Key Features

- Real-time comment extraction via YouTube API  
- Chrome extension for user-friendly interface  
- Preprocessing handles emojis, slang, Hinglish  
- Multiple ML models with iterative improvements  
- Dockerized Flask API deployed on AWS EC2  
- Experiment tracking & model registry via MLflow  
- CI/CD automation with GitHub Actions  

---

## 🏗 MLOps & Deployment

- **Experiment Tracking:** MLflow  
- **Model Registry:** Staging → Production promotion  
- **Containerization:** Docker  
- **Cloud Hosting:** AWS EC2  
- **Automated Testing & CI/CD:** GitHub Actions  
- **Scalable API:** Flask REST endpoint  

---

## 📊 Visual Insights

_Visual placeholders — add actual charts for full portfolio effect:_  

- Confusion Matrix  
- Sentiment Distribution Graph  
- Word Cloud  
- System Architecture Diagram  

---

## 📁 Frontend (Chrome Extension)

- `popup.html` – User interface popup  
- `popup.js` – JS logic for fetching and displaying sentiment  
- `styles.css` – Extension styling  

---

## ⚙️ Tech Stack

Python | Flask | LightGBM | TF-IDF | NLTK | Docker | AWS EC2 | MLflow | GitHub Actions | Chrome Extension | Pandas | Matplotlib

---

## 📌 Repository & Portfolio

- GitHub: [https://github.com/MuktiKsinha/Youtube-sentiment-analysis](https://github.com/MuktiKsinha/Youtube-sentiment-analysis)  
- Portfolio: `[Add Portfolio URL]`  

---

## 📈 Impact

- Automates YouTube comment sentiment analysis  
- Provides actionable insights for content creators and marketers  
- Demonstrates full ML lifecycle: experimentation → deployment → monitoring  
- Optimized for real-world noisy social media data  

---
