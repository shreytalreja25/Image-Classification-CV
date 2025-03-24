import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import os
from tqdm import tqdm
import time
from datetime import datetime
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

def train_resnet(train_loader, num_classes, num_epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n🖥️  Training on: {device} ({device_name})")

    if device.type == "cuda":
        print(f"🧠 CUDA Memory Allocated: {torch.cuda.memory_allocated() // (1024 ** 2)} MB")

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze feature extractor

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"🔁 Epoch {epoch+1}/{num_epochs}", ncols=100)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"\n✅ Epoch {epoch+1} completed — Avg Loss: {avg_loss:.4f} ⏱️ {epoch_time:.2f}s")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_finetuned.pth")
    print("💾 Model saved to: models/resnet18_finetuned.pth")

    return model


def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    metrics = evaluate_classification(all_labels, all_preds, class_names, verbose=False)

    # Timestamp & folder
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = f"results/DL_results/resnet18_eval_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # Save report to folder
    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write(f"Model: ResNet-18 (fine-tuned)\n")
        f.write(f"Device: {device}\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)

    print(f"\n✅ Classification report saved to: {report_file}")

    # Save confusion matrix
    cm_path = os.path.join(result_dir, "confmat.png")
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        normalize=True,
        title="ResNet-18 Confusion Matrix",
        save_path=cm_path
    )
    print(f"🖼️ Confusion matrix saved to: {cm_path}")

    # Also print to terminal
    print("\n📊 Classification Report:")
    print(report)
