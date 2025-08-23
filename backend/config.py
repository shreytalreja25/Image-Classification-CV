from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Development Mode
    dev_mode: bool = True
    
    # Backend URLs
    local_backend_url: str = "http://localhost:8000"
    production_backend_url: str = "https://your-backend.onrender.com"
    
    # Frontend URLs
    local_frontend_url: str = "http://localhost:3000"
    production_frontend_url: str = "https://your-frontend.vercel.app"
    
    # MongoDB Configuration
    mongodb_uri: str = "mongodb+srv://username:password@cluster.mongodb.net/aerial_classification?retryWrites=true&w=majority"
    mongodb_db: str = "aerial_classification"
    
    # JWT Configuration
    jwt_secret_key: str = "your-super-secret-jwt-key-here"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Model and Dataset Paths
    model_path: str = "./models/"
    dataset_path: str = "./subset/"
    
    # CORS Configuration - Store as string and parse to list
    allowed_origins: str = "http://localhost:3000,https://your-frontend.vercel.app"
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    @property
    def backend_url(self) -> str:
        """Get the appropriate backend URL based on dev mode"""
        return self.local_backend_url if self.dev_mode else self.production_backend_url
    
    @property
    def frontend_url(self) -> str:
        """Get the appropriate frontend URL based on dev mode"""
        return self.local_frontend_url if self.dev_mode else self.production_backend_url
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins based on dev mode"""
        if self.dev_mode:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
            ]
        # Parse the comma-separated string into a list
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.model_path, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
