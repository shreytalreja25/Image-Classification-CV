from dl_methods.efficientnet import get_data_loaders, evaluate_model
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import os

print("Loading EfficientNet-B0 model for evaluation only...")

# Load data
_, test_loader, class_names = get_data_loaders('data', batch_size=32)

# Load model structure and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name('efficientnet-b0')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, len(class_names))
model.load_state_dict(torch.load("models/efficientnet_b0.pth", map_location=device))

# Evaluate
evaluate_model(model, test_loader, class_names)
