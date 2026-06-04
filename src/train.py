import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from config import settings

def train_pipeline():
    print("🚀 Starting Production Training Pipeline...")
    
    if not os.path.exists(settings.DATA_PATH):
        raise FileNotFoundError(f"Missing training dataset at: {settings.DATA_PATH}")
        
    print("Loading mixed Arabic-English dataset...")
    df = pd.read_csv(settings.DATA_PATH)
    
    X = df["clean_text"]
    y = df["sentiment"]
    
    # Split data to validate model health
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing language text strings...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    print("Fitting Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    val_accuracy = model.score(X_val_vec, y_val)
    print(f"📊 Validation Accuracy Score: {val_accuracy * 100:.2f}%")
    
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    
    joblib.dump(model, settings.MODEL_PATH)
    joblib.dump(vectorizer, settings.VECTORIZER_PATH)
    print("💾 Model & Vectorizer saved independently as production artifacts!")

if __name__ == "__main__":
    train_pipeline()