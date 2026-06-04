import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_TITLE: str = "Arabic-English Social Sentiment Analyzer"
    
    # Project Base Directory
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Strictly Configured Data and Model Paths
    DATA_PATH: str = os.path.join(BASE_DIR, "data", "clean_data.csv")
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "model.pkl")
    VECTORIZER_PATH: str = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

    class Config:
        env_file = ".env"

settings = Settings()