# 🎬 YouTube Sentiment Analysis

Analyze YouTube video comments in real-time and classify them into **Positive, Negative, or Neutral** sentiments.  
The project includes **multiple ML experimentation cycles**, **MLOps integration**, and **deployment on AWS EC2** with a Dockerized backend and Chrome extension frontend.

---

## 🏢 Business Context

As an **Influencer Management Company** focused on expanding our creator network, we aim to attract more influencers to join our platform. However, due to a **limited marketing budget**, traditional advertising and paid outreach are not viable options.

To overcome this constraint, we identified a **key pain point** for influencers:  
👉 Understanding audience sentiment and feedback on their YouTube content.

Creators often struggle to manually sift through thousands of comments to assess:

- Are viewers responding positively or negatively?
- Which videos drive the strongest engagement?
- What kind of content should they produce more of?

By addressing this challenge directly, we aim to:

✔ Provide influencers with **instant, data-driven insights** into their audience  
✔ Increase creator engagement with our platform  
✔ Position our company as a **trusted analytics partner**  
✔ Boost onboarding and long-term retention without high marketing costs

> 🎯 Strategic Goal: Deliver value **first**, attract influencers **organically**, and scale sustainably.


## 📂 Project Folder Structure

<details>
<summary><strong>Click to expand</strong></summary>

📦 YouTube-sentiment-analysis

├─ 📁 data
 # Raw and processed data
├─ 📁 flask_app
 # Backend API for sentiment prediction
│ ├─ 🧩 app.py

│ └─ 🧩 utils.py

├─ 📁 frontend
 # Chrome extension / UI
│ ├─ 🌐 popup.html

│ ├─ 🎛️ popup.js

│ └─ 🎨 styles.css

├─ 📁 models
 # Trained ML models + vectorizers
├─ 📁 mlflow
 # Experiment tracking + model registry
├─ 📁 notebooks
 # Jupyter experimentation
├─ 📁 scripts
 # Automation + data pipeline scripts
├─ 📁 src
 # Core Python modules
├─ 🐳 Dockerfile
 # Docker config for deployment
├─ 📄 requirements.txt
 # Python dependencies
├─ 📄 README.md
 # Documentation
└─ ⚖️ LICENSE
 # Project license



</details>

---

## 🧪 Experiments & Model Performance

Multiple experimentation cycles improved model performance from **49% → 86% F1-score**.  
All experiments are tracked via **MLflow** with full reproducibility.

| Version | Model | Vectorizer / Technique | Accuracy | Macro Precision | Macro Recall | F1-score | Notes |
|--------:|-------|----------------------|---------:|----------------:|-------------:|---------:|------|
| v1 | Random Forest | Bag of Words (BoW) | 66% | 76% | 56% | 49% | Baseline |
| v2 | Random Forest | BoW + Trigrams | 65% | 75% | 57% | 52% | Trigram features tested |
| v3 | Random Forest | BoW + max_features | 66% | 76% | 57% | 51% | max_features=5000 performed best |
| v4 | Random Forest | Class imbalance handling | 66% | 68% | 66% | 65% | Undersampling improved recall |
| v5 | RF / XGB / SVM / NB / LR / LGBM / KNN | Model selection | 78% | 79% | 78% | 76% | LightGBM selected as best |
| v6 | LightGBM | Hyperparameter tuning (Optuna) | 78% | 77% | 77% | 76% | 100+ trials on HP tuning |
| ⭐ v7 (Final) | **LightGBM** | **BoW + n-grams + tuned parameters** | **87%** | **86%** | **86%** | **86%** | Best performance combining all improvements |

**Final Model Metrics (Macro Avg):**
- **Precision:** 0.86
- **Recall:** 0.86
- **F1-score:** 0.86  

📈 **Overall Performance Improvement Highlights**

✔ **Accuracy:** 66% → 87% (**+21% increase**)  
✔ **Precision:** 76% → 86% (**+10% increase**)  
✔ **Recall:** 56% → 86% (**+30% increase**)  
✔ **F1-score:** 49% → 86% (**+37% increase**) 🚀  

> Huge jump in model reliability due to **hyperparameter tuning**,  
> **n-gram features**, and **class imbalance handling**.
---

## 🚀 Key Features

- Real-time YouTube comment extraction via YouTube API  
- Chrome extension for user-friendly interface  
- Preprocessing handles emojis, slang, Hinglish  
- Multiple ML models with iterative improvements  
- Dockerized Flask API deployed on AWS EC2  
- Experiment tracking & model registry via MLflow  
- CI/CD automation with GitHub Actions  

---

## 🏗 MLOps & Deployment

| Component | Tool |
|----------|------|
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| CI/CD Automation | GitHub Actions |
| API Hosting | AWS EC2 |
| Containerization | Docker |
| Monitoring | Logs + MLflow Metrics |
| Serving Pattern | Flask REST API |

Deployment Workflow:  
**Dev → MLflow → Docker → CI/CD → AWS EC2 (Production)**

---


## 📊 Visual Insights

To showcase the model’s performance and system design effectively, the following visualizations are included (or can be added for portfolio enhancement):

---

### 🔁 Confusion Matrix
Displays correct vs incorrect predictions across sentiment classes.

📌 <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/975a7891-a500-47bb-b453-2c12ff9fade6" />

---


---

### 📈 Sentiment Distribution Graph
Shows how sentiments are distributed across all extracted YouTube comments.

📌 <img width="400" height="666" alt="toutube_plugin" src="https://github.com/user-attachments/assets/808d1372-d8cc-4a75-a83b-3d5f95e004e3" />
   <img width="334" height="602" alt="yt2" src="https://github.com/user-attachments/assets/c7095c7b-3eef-433e-bf1b-781d95ea4b8d" />
   <img width="332" height="592" alt="yt3" src="https://github.com/user-attachments/assets/767b3478-1cc5-4832-873c-4ca8acd74824" />



---

### 📒 MLflow Experiment Dashboard
Experiment tracking with metrics, artifacts, and model versioning.

📌 <img width="1518" height="728" alt="image" src="https://github.com/user-attachments/assets/566d1a8f-5f0a-443c-af64-0f74c3987347" />


---

# High-Level End-to-End Flow

YouTube API → ML Model → Flask API → Chrome Extension → User

# High-Level End-to-End Flow

YouTube API → ML Model → Flask API → Chrome Extension → User

```
┌──────────────────────────────────────────────┐
│ 🧑‍💻 YouTube User                             │
│ (Chrome Browser / Extension)                  │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│ 📡 YouTube Data API v3                        │
│ (Fetches video comments)                      │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│ 🎨 Chrome Extension Frontend                  │
│ - HTML / CSS / JS                             │
│ - Cleans comments                             │
│ - Sends requests to Flask API                 │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│ 🐍 Flask REST API                              │
│ - Dockerized on AWS EC2                        │
│ - Receives comments & performs inference      │
│ - Communicates with MLflow                     │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ 🧮 Bag-of-Words Vectorizer     │   │ 🌟 LightGBM Model             │
│ - Converts text → sparse vec   │─▶ │ - Predicts sentiment         │
└───────────────────────────────┘   └───────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│ 🔎 MLflow Tracking Server                     │
│ - Stores metrics, params, and artifacts      │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│ 📊 Sentiment Output to Chrome Extension UI   │
│ - Positive / Neutral / Negative              │
└──────────────────────────────────────────────┘
```

## ☁️ Deployment Architecture (AWS + Docker + CI/CD)

```
┌─────────────────────────────────────┐
│ 👩‍💻 Developer (GitHub Repo Push)   │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 🔄 GitHub Actions CI/CD Pipeline    │
│ • Run Tests (pytest)                │
│ • Validate model signature          │
│ • Build & Push Docker Image         │
│ • Auto-deploy to EC2 (SSH / Docker) │
└─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 🐳 Dockerized Flask API              │
│ Hosted on AWS EC2                    │
│ • Model & Vectorizer mounted         │
│ • REST endpoint for predictions      │
└──────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 🧑‍💻 Chrome Extension Frontend        │
│ Calls Flask API for predictions      │
└──────────────────────────────────────┘
```
## 🔁 MLflow Experiment Lifecycle

```
┌──────────────────────────────────────┐
│ 📂 Data Collection (YouTube API)     │
│ Raw comments                         │
└──────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 🧹 Preprocessing & Vectorization     │
│ • Hinglish cleaning                  │
│ • Emoji & slang handling             │
│ • BoW Vectorizer                     │
└──────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ 🧠 Model Training                    │
│ LightGBM + Hyperparameter Tuning     │
└──────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 📊 MLflow Tracking Server                   │
│ Logs: Metrics | Parameters | Artifacts      │
│ Multiple experiment versions (v1 → v6⭐)     │
└─────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 🔐 MLflow Model Registry                     │
│ • Staging → Production promotion pipeline    │
└─────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 🐳 Deploy Production Model via Docker        │
│ Auto-reload on new production release        │
└─────────────────────────────────────────────┘
```






## 📁 Frontend (Chrome Extension)

- `popup.html` – UI layout  
- `popup.js` – API communication logic  
- `styles.css` – Styling for popup  

---

## ⚙️ Tech Stack

Python • Flask • LightGBM • Bag-of-Words + N-grams  
NLTK • AWS EC2 • Docker • MLflow • GitHub Actions  
Chrome Extension • Pandas • Matplotlib

---

## 📌 Repository & Portfolio

- GitHub: https://github.com/MuktiKsinha/Youtube-sentiment-analysis  
- Portfolio: _[Add Portfolio URL]_  

---

## 📈 Impact

- Automates YouTube comment sentiment analysis  
- Supports creators and marketing analytics  
- Full ML lifecycle: data → model → deployment → monitoring  
- Optimized for real-world noisy social media text

---

