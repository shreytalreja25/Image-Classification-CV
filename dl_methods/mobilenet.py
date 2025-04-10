import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
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

def train_mobilenet(train_loader, num_classes, num_epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False  # freeze backbone

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f} ⏱️ {time.time()-start:.2f}s")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mobilenet_v2.pth")
    print("Model saved to: models/mobilenet_v2.pth")
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
    result_dir = f"results/DL_results/mobilenet_eval_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write("Model: MobileNetV2\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)

    print(f"\n✅ Classification report saved to: {result_dir}/report.txt")

    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        normalize=True,
        title="MobileNetV2 Confusion Matrix",
        save_path=os.path.join(result_dir, "confmat.png")
    )
    print(f"Confusion matrix saved to: {result_dir}/confmat.png")
    print(report)
