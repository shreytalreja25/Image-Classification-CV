# main_mobilenet_eval_only.py
import torch
from torchvision import models
import torch.nn as nn
from dl_methods.mobilenet import get_data_loaders, evaluate_model

print("\n🧪 Loading MobileNetV2 model for evaluation...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model structure
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 15)  # 15 classes
model.load_state_dict(torch.load("models/mobilenet_v2.pth", map_location=device))
model.to(device)

# Load data
_, test_loader, class_names = get_data_loaders("data")

# Evaluate
evaluate_model(model, test_loader, class_names)
