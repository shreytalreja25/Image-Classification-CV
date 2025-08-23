from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import certifi
from typing import Optional, List
import asyncio
from datetime import datetime
from config import settings

class Database:
    client: Optional[AsyncIOMotorClient] = None
    sync_client: Optional[MongoClient] = None

async def connect_to_mongo():
    """Create database connection and verify connectivity with a ping."""
    try:
        ca_bundle_path = certifi.where()
        Database.client = AsyncIOMotorClient(
            settings.mongodb_uri,
            tls=True,
            tlsCAFile=ca_bundle_path,
        )
        Database.sync_client = MongoClient(
            settings.mongodb_uri,
            tls=True,
            tlsCAFile=ca_bundle_path,
        )
        # Attempt a quick ping using the sync client to fail fast on SSL/creds issues
        Database.sync_client.admin.command("ping")
        print("Connected to MongoDB.")
    except PyMongoError as exc:
        # Keep app running; downstream features that need DB should handle None
        print(f"MongoDB connection failed: {exc}")

async def close_mongo_connection():
    """Close database connection."""
    if Database.client:
        Database.client.close()
    if Database.sync_client:
        Database.sync_client.close()
    print("Disconnected from MongoDB.")

def get_database():
    """Get database instance."""
    return Database.client[settings.mongodb_db]

def get_sync_database():
    """Get synchronous database instance."""
    return Database.sync_client[settings.mongodb_db]

# Database collections
async def get_collection(collection_name: str):
    """Get a specific collection from the database."""
    db = get_database()
    return db[collection_name]

def get_sync_collection(collection_name: str):
    """Get a specific collection from the database synchronously."""
    db = get_sync_database()
    return db[collection_name]

# Sample data for testing
async def initialize_sample_data():
    """Initialize sample data for the application."""
    try:
        db = get_database()
        # If DB is not connected, get_database() will raise; handle below
        # Check if data already exists
        if await db.training_stats.count_documents({}) > 0:
            return

        # Sample training statistics
        sample_stats = [
        {
            "model_name": "EfficientNet-B0",
            "accuracy": 95.46,
            "macro_f1": 95.45,
            "precision": 95.53,
            "recall": 95.33,
            "timestamp": datetime.now(),
            "training_duration": "2h 15m",
            "dataset_size": "12,000 images",
            "categories": 15,
            "status": "completed"
        },
        {
            "model_name": "MobileNetV2",
            "accuracy": 93.00,
            "macro_f1": 93.02,
            "precision": 93.10,
            "recall": 93.00,
            "timestamp": datetime.now(),
            "training_duration": "1h 45m",
            "dataset_size": "12,000 images",
            "categories": 15,
            "status": "completed"
        },
        {
            "model_name": "ResNet-18",
            "accuracy": 90.87,
            "macro_f1": 90.83,
            "precision": 91.00,
            "recall": 90.87,
            "timestamp": datetime.now(),
            "training_duration": "2h 30m",
            "dataset_size": "12,000 images",
            "categories": 15,
            "status": "completed"
        }
    ]

        await db.training_stats.insert_many(sample_stats)
        print("Sample data initialized.")
    except Exception as exc:
        # Do not block app startup if sample data cannot be created
        print(f"Skipping sample data initialization due to DB error: {exc}")

# Database models
class TrainingStats:
    def __init__(self, collection):
        self.collection = collection
    
    async def get_all_stats(self):
        """Get all training statistics."""
        cursor = self.collection.find({}).sort("timestamp", -1)
        stats = await cursor.to_list(length=100)
        return stats
    
    async def get_model_stats(self, model_name: str):
        """Get statistics for a specific model."""
        return await self.collection.find_one({"model_name": model_name})
    
    async def add_training_stats(self, stats_data: dict):
        """Add new training statistics."""
        stats_data["timestamp"] = datetime.now()
        result = await self.collection.insert_one(stats_data)
        return result.inserted_id
    
    async def update_training_status(self, model_name: str, status: str):
        """Update training status for a model."""
        return await self.collection.update_one(
            {"model_name": model_name},
            {"$set": {"status": status}}
        )

class ModelPredictions:
    def __init__(self, collection):
        self.collection = collection
    
    async def save_prediction(self, prediction_data: dict):
        """Save a model prediction."""
        prediction_data["timestamp"] = datetime.now()
        result = await self.collection.insert_one(prediction_data)
        return result.inserted_id
    
    async def get_recent_predictions(self, limit: int = 50):
        """Get recent predictions."""
        cursor = self.collection.find({}).sort("timestamp", -1).limit(limit)
        predictions = await cursor.to_list(length=limit)
        return predictions
    
    async def get_predictions_by_model(self, model_name: str, limit: int = 20):
        """Get predictions by model name."""
        cursor = self.collection.find({"model_name": model_name}).sort("timestamp", -1).limit(limit)
        predictions = await cursor.to_list(length=limit)
        return predictions
