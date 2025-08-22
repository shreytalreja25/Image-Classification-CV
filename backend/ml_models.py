import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from config import settings

# Model class definitions
class EfficientNetModel:
    def __init__(self):
        self.model = None
        self.transform = None
        self.class_names = [
            'Agriculture', 'Airport', 'Beach', 'City', 'Desert',
            'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
            'Parking', 'Port', 'Railway', 'Residential', 'River'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained EfficientNet model"""
        try:
            # Load the trained model
            model_path = os.path.join(settings.model_path, "efficientnet_model.pth")
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
            else:
                # Use pre-trained EfficientNet as fallback
                from efficientnet_pytorch import EfficientNet
                self.model = EfficientNet.from_pretrained('efficientnet-b0')
                self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error loading EfficientNet model: {e}")
            self.model = None
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on an image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get results
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()
        
        # Get all predictions
        all_predictions = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        processing_time = time.time() - start_time
        
        return predicted_class, confidence, all_predictions, processing_time

class MobileNetModel:
    def __init__(self):
        self.model = None
        self.transform = None
        self.class_names = [
            'Agriculture', 'Airport', 'Beach', 'City', 'Desert',
            'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
            'Parking', 'Port', 'Railway', 'Residential', 'River'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained MobileNet model"""
        try:
            # Load the trained model
            model_path = os.path.join(settings.model_path, "mobilenet_model.pth")
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
            else:
                # Use pre-trained MobileNet as fallback
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error loading MobileNet model: {e}")
            self.model = None
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on an image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get results
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()
        
        # Get all predictions
        all_predictions = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        processing_time = time.time() - start_time
        
        return predicted_class, confidence, all_predictions, processing_time

class ResNetModel:
    def __init__(self):
        self.model = None
        self.transform = None
        self.class_names = [
            'Agriculture', 'Airport', 'Beach', 'City', 'Desert',
            'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
            'Parking', 'Port', 'Railway', 'Residential', 'River'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained ResNet model"""
        try:
            # Load the trained model
            model_path = os.path.join(settings.model_path, "resnet_model.pth")
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
            else:
                # Use pre-trained ResNet as fallback
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error loading ResNet model: {e}")
            self.model = None
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on an image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get results
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()
        
        # Get all predictions
        all_predictions = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        processing_time = time.time() - start_time
        
        return predicted_class, confidence, all_predictions, processing_time

class TraditionalMLModel:
    def __init__(self):
        self.models = {}
        self.class_names = [
            'Agriculture', 'Airport', 'Beach', 'City', 'Desert',
            'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
            'Parking', 'Port', 'Railway', 'Residential', 'River'
        ]
        self.load_models()
    
    def load_models(self):
        """Load traditional ML models"""
        try:
            # Load SIFT + Random Forest
            rf_path = os.path.join(settings.model_path, "rf_model.pkl")
            if os.path.exists(rf_path):
                self.models['random_forest'] = joblib.load(rf_path)
            
            # Load SIFT + SVM
            svm_path = os.path.join(settings.model_path, "svm_model.pkl")
            if os.path.exists(svm_path):
                self.models['svm'] = joblib.load(svm_path)
                
        except Exception as e:
            print(f"Error loading traditional ML models: {e}")
    
    def extract_sift_features(self, image_path: str) -> np.ndarray:
        """Extract SIFT features from image"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is not None:
                # Use mean of descriptors as feature vector
                return np.mean(descriptors, axis=0)
            else:
                # Return zero vector if no features found
                return np.zeros(128)
        except Exception as e:
            print(f"Error extracting SIFT features: {e}")
            return np.zeros(128)
    
    def predict(self, model_name: str, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction using traditional ML model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        start_time = time.time()
        
        # Extract features
        features = self.extract_sift_features(image_path)
        features = features.reshape(1, -1)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        predicted_class = self.class_names[prediction]
        confidence = np.max(probabilities)
        
        # Get all predictions
        all_predictions = {
            self.class_names[i]: prob 
            for i, prob in enumerate(probabilities)
        }
        
        processing_time = time.time() - start_time
        
        return predicted_class, confidence, all_predictions, processing_time

# Model factory
class ModelFactory:
    def __init__(self):
        self.models = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all available models"""
        try:
            self.models['EfficientNet-B0'] = EfficientNetModel()
            self.models['MobileNetV2'] = MobileNetModel()
            self.models['ResNet-18'] = ResNetModel()
            self.models['Random Forest'] = TraditionalMLModel()
            print("All models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {e}")
    
    def get_model(self, model_name: str):
        """Get a specific model by name"""
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not available")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def predict(self, model_name: str, image_path: str) -> Tuple[str, float, Dict[str, float], float]:
        """Make prediction using specified model"""
        model = self.get_model(model_name)
        
        if hasattr(model, 'predict'):
            if model_name == 'Random Forest':
                return model.predict('random_forest', image_path)
            else:
                return model.predict(image_path)
        else:
            raise ValueError(f"Model {model_name} does not support prediction")

# Global model factory instance
model_factory = ModelFactory()
