from dl_methods.efficientnet import get_data_loaders, train_efficientnet, evaluate_model

print("⚡ Starting EfficientNet-B0 training...")

train_loader, test_loader, class_names = get_data_loaders('data', batch_size=32)
model = train_efficientnet(train_loader, num_classes=len(class_names), num_epochs=5)
evaluate_model(model, test_loader, class_names)
