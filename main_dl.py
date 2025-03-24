from dl_methods.resnet import get_data_loaders, train_resnet, evaluate_model

print("🧠 Starting Deep Learning Pipeline...")

train_loader, test_loader, class_names = get_data_loaders('data', batch_size=32)
model = train_resnet(train_loader, num_classes=len(class_names), num_epochs=5)
evaluate_model(model, test_loader, class_names)
