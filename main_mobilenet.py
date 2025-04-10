# main_mobilenet.py
from dl_methods.mobilenet import get_data_loaders, train_mobilenet, evaluate_model

print("\n🚀 Starting MobileNetV2 training...")

data_dir = "data"
train_loader, test_loader, class_names = get_data_loaders(data_dir)
model = train_mobilenet(train_loader, num_classes=len(class_names))
evaluate_model(model, test_loader, class_names)
