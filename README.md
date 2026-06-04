# Arabic-English Social Sentiment Analyzer 🚀

A production-grade, asynchronous Natural Language Processing (NLP) API designed to classify text sentiment across English and Arabic social media data. Built with **FastAPI**, **Pydantic Validation**, and **Scikit-Learn**, this project models enterprise-level backend practices optimized for scalability and zero-downtime deployment.

---

## 🎯 Core Technical Features

* **Asynchronous High-Performance Architecture:** Utilizes FastAPI's `async/await` runtime handlers to process concurrent data pipeline inference frames cleanly.
* **Robust Configuration Layer:** Outfitted with `pydantic-settings` to decouple environment contexts securely from application engines.
* **Cross-Language Text Representation:** Employs an optimized `TF-IDF Vectorizer` supporting single words and bigrams (`ngram_range=(1,2)`) to accurately capture structural contexts in mixed English prose and Arabic dialects.
* **Production-Grade Security Foundations:** Pre-configured with Cross-Origin Resource Sharing (`CORS`) middleware layer mechanics allowing smooth frontend connectivity out-of-the-box.

---

## 📂 Project Architecture

```text
social-sentiment-analyzer/
├── data/
│   └── clean_data.csv          # Local mixed training data repository
├── models/
│   ├── model.pkl               # Trained Classifier artifact binary 
│   └── vectorizer.pkl          # Extracted Text Vocabulary mapping matrix
├── src/
│   ├── config.py               # Decentralized Pydantic Settings management
│   ├── main.py                 # Async FastAPI Backend Router app engine
│   └── train.py                # Standalone feature-extraction ML training pipeline
├── requirements.txt            # Environment blueprint dependencies 
└── README.md                   # System design & run documentation