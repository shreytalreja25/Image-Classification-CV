import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from datetime import datetime
import os
from utils.metrics import evaluate_classification, plot_confusion_matrix

def get_data_loaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.classes

def train_efficientnet(train_loader, num_classes, num_epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = EfficientNet.from_pretrained('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/efficientnet_b0.pth")
    print("Model saved to: models/efficientnet_b0.pth")

    return model

def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    metrics = evaluate_classification(all_labels, all_preds, class_names, verbose=False)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join("results", "DL_results", f"efficientnet_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write("Model: EfficientNet-B0 (fine-tuned)\n")
        f.write(f"Device: {device}\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)

    print(f"Report saved to: {report_file}")

    cm_path = os.path.join(result_dir, "confmat.png")
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        normalize=True,
        title="EfficientNet-B0 Confusion Matrix",
        save_path=cm_path
    )

    print(f"Confusion matrix saved to: {cm_path}")
    print("\nClassification Report (EfficientNet):")
    print(report)
