from dl_methods.resnet import get_data_loaders, evaluate_model
from torchvision import models
import torch
import torch.nn as nn

print("🧪 Loading model for evaluation only...")

# Load data
_, test_loader, class_names = get_data_loaders('data', batch_size=32)

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Match your class count
model.load_state_dict(torch.load("models/resnet18_finetuned.pth", map_location=device))

# Evaluate
evaluate_model(model, test_loader, class_names)
