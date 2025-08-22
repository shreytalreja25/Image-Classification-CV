from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import uuid
import os
import random
from datetime import datetime
from typing import List, Dict, Any
import asyncio

from config import settings
from database import connect_to_mongo, close_mongo_connection, get_collection, TrainingStats, ModelPredictions
from models import (
    PredictionRequest, PredictionResponse, TrainingStatsResponse, 
    TestImageRequest, TestImageResponse, DashboardStats, SuccessResponse, ErrorResponse
)
from ml_models import model_factory

# Create FastAPI app
app = FastAPI(
    title="Aerial Landscape Classification API",
    description="API for classifying aerial landscape images using various ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                continue

manager = ConnectionManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    # Initialize sample data
    from database import initialize_sample_data
    await initialize_sample_data()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Get available models
@app.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available models"""
    try:
        models = model_factory.get_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get model information
@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model = model_factory.get_model(model_name)
        
        # Get training stats from database
        training_stats_collection = await get_collection("training_stats")
        stats = await training_stats_collection.find_one({"model_name": model_name})
        
        model_info = {
            "name": model_name,
            "description": f"Trained {model_name} model for aerial landscape classification",
            "architecture": model_name,
            "parameters": "Pre-trained with fine-tuning",
            "input_size": "224x224 RGB",
            "accuracy": stats["accuracy"] if stats else 0.0,
            "speed": "Fast inference",
            "status": stats["status"] if stats else "unknown"
        }
        
        return model_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

# Make prediction
@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make prediction using specified model"""
    try:
        # For now, we'll use a sample image path
        # In production, you'd handle file uploads or URLs
        sample_image_path = os.path.join(settings.dataset_path, "test", "Agriculture", "001.jpg")
        
        if not os.path.exists(sample_image_path):
            raise HTTPException(status_code=404, detail="Sample image not found")
        
        # Make prediction
        predicted_class, confidence, all_predictions, processing_time = model_factory.predict(
            request.model_name, sample_image_path
        )
        
        # Create prediction response
        prediction_response = PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            model_name=request.model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=all_predictions,
            processing_time=processing_time,
            timestamp=datetime.now(),
            image_path=sample_image_path
        )
        
        # Save prediction to database
        predictions_collection = await get_collection("predictions")
        await predictions_collection.insert_one(prediction_response.dict())
        
        return prediction_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Get test image
@app.post("/test-image", response_model=TestImageResponse)
async def get_test_image(request: TestImageRequest):
    """Get a random test image for testing"""
    try:
        test_dir = os.path.join(settings.dataset_path, "test")
        
        if request.random:
            # Get random category
            categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            category = random.choice(categories)
        else:
            category = request.category or random.choice([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        
        category_path = os.path.join(test_dir, category)
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        
        if not images:
            raise HTTPException(status_code=404, detail=f"No images found in category {category}")
        
        # Select random image
        selected_image = random.choice(images)
        image_path = os.path.join(category_path, selected_image)
        
        return TestImageResponse(
            image_path=image_path,
            category=category,
            filename=selected_image
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test image: {str(e)}")

# Get training statistics
@app.get("/training-stats", response_model=List[TrainingStatsResponse])
async def get_training_stats():
    """Get all training statistics"""
    try:
        training_stats_collection = await get_collection("training_stats")
        stats = await training_stats_collection.find({}).to_list(length=100)
        
        # Convert ObjectId to string for JSON serialization
        for stat in stats:
            stat["_id"] = str(stat["_id"])
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")

# Get dashboard statistics
@app.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        db = await get_collection("training_stats")
        
        # Get model performance
        model_performance = await db.find({}).to_list(length=100)
        for stat in model_performance:
            stat["_id"] = str(stat["_id"])
        
        # Get recent predictions
        predictions_collection = await get_collection("predictions")
        recent_predictions = await predictions_collection.find({}).sort("timestamp", -1).limit(10).to_list(length=10)
        for pred in recent_predictions:
            pred["_id"] = str(pred["_id"])
        
        # Calculate summary stats
        total_models = len(model_performance)
        total_predictions = await predictions_collection.count_documents({})
        
        if model_performance:
            average_accuracy = sum(stat["accuracy"] for stat in model_performance) / len(model_performance)
            best_model = max(model_performance, key=lambda x: x["accuracy"])["model_name"]
        else:
            average_accuracy = 0.0
            best_model = "None"
        
        return DashboardStats(
            total_models=total_models,
            total_predictions=total_predictions,
            average_accuracy=average_accuracy,
            best_model=best_model,
            recent_predictions=recent_predictions,
            model_performance=model_performance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard stats: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send heartbeat
            await asyncio.sleep(settings.ws_heartbeat_interval)
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Broadcast training progress
async def broadcast_training_progress(model_name: str, epoch: int, total_epochs: int, loss: float, accuracy: float):
    """Broadcast training progress to all connected WebSocket clients"""
    message = {
        "type": "training_progress",
        "data": {
            "model_name": model_name,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
    }
    await manager.broadcast(json.dumps(message))

# Get image file
@app.get("/images/{category}/{filename}")
async def get_image(category: str, filename: str):
    """Serve image files"""
    try:
        image_path = os.path.join(settings.dataset_path, "test", category, filename)
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            message=str(exc.detail),
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
