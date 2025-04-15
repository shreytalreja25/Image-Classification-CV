# dl_methods/densenet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils.metrics import evaluate_classification, plot_confusion_matrix
from datetime import datetime
from tqdm import tqdm

def run_densenet(
    train_dir: str,
    test_dir: str,
    num_classes: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    save_model_path: str,
    results_root: str,
    device: str = None
):
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Datasets & loaders
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=80)
        running_loss = 0.0
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/loop.n)

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    class_names = train_ds.classes
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📊 DenseNet-121 Classification Report:\n", report)

    # Metrics & plots
    metrics = evaluate_classification(y_true, y_pred, class_names, verbose=False)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join(results_root, f"densenet121_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save report
    with open(os.path.join(result_dir, "report.txt"), "w") as f:
        f.write(f"Timestamp: {timestamp}\nModel: DenseNet-121\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nFull Report:\n")
        f.write(report)

    # Save confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=True,
        title="DenseNet-121 Confusion Matrix",
        save_path=os.path.join(result_dir, "confmat.png")
    )
    print(f"\n✅ Results saved to {result_dir}")

    # Save model weights
    torch.save(model.state_dict(), save_model_path)
    print(f"💾 Model weights saved to {save_model_path}")
