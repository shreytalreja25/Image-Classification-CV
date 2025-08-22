from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    EFFICIENTNET = "EfficientNet-B0"
    MOBILENET = "MobileNetV2"
    RESNET = "ResNet-18"
    DENSENET = "DenseNet-121"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class PredictionRequest(BaseModel):
    model_name: ModelType
    image_url: Optional[str] = None
    image_file: Optional[str] = None  # Base64 encoded image

class PredictionResponse(BaseModel):
    prediction_id: str
    model_name: str
    predicted_class: str
    confidence: float
    all_predictions: Dict[str, float]
    processing_time: float
    timestamp: datetime
    image_path: Optional[str] = None

class TrainingStatsResponse(BaseModel):
    model_name: str
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    timestamp: datetime
    training_duration: str
    dataset_size: str
    categories: int
    status: TrainingStatus

class ModelInfo(BaseModel):
    name: str
    description: str
    architecture: str
    parameters: str
    input_size: str
    accuracy: float
    speed: str
    status: TrainingStatus

class TestImageRequest(BaseModel):
    category: Optional[str] = None
    random: bool = True

class TestImageResponse(BaseModel):
    image_path: str
    category: str
    filename: str
    model_predictions: Optional[Dict[str, float]] = None

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class TrainingProgress(BaseModel):
    model_name: str
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    status: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# API Response Models
class DashboardStats(BaseModel):
    total_models: int
    total_predictions: int
    average_accuracy: float
    best_model: str
    recent_predictions: List[PredictionResponse]
    model_performance: List[TrainingStatsResponse]

class ModelComparison(BaseModel):
    models: List[str]
    metrics: Dict[str, List[float]]
    categories: List[str]
    category_performance: Dict[str, Dict[str, float]]

# WebSocket Event Types
class WebSocketEventType(str, Enum):
    TRAINING_PROGRESS = "training_progress"
    PREDICTION_UPDATE = "prediction_update"
    MODEL_STATUS_UPDATE = "model_status_update"
    SYSTEM_NOTIFICATION = "system_notification"
    HEARTBEAT = "heartbeat"
