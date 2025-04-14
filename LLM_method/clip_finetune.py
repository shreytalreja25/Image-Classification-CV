import os
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
from PIL import Image
import multiprocessing

def collate_fn(batch):
    # batch: List[(PIL.Image, int)]
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def main():
    # ────────────────────────────────
    # Suppress HF warnings
    # ────────────────────────────────
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]   = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ────────────────────────────────
    # Config
    # ────────────────────────────────
    DATA_PATH  = "archive/Aerial_Landscapes"
    MODEL_NAME = "openai/clip-vit-base-patch32"
    SAVE_DIR   = "clip_model"
    BATCH_SIZE = 8
    EPOCHS     = 4
    LR         = 1e-4

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ────────────────────────────────
    # Load CLIP & freeze
    # ────────────────────────────────
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip      = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    for p in clip.parameters():
        p.requires_grad = False

    # ────────────────────────────────
    # Classifier head
    # ────────────────────────────────
    image_dim   = clip.config.projection_dim
    num_classes = len(os.listdir(DATA_PATH))
    classifier  = nn.Linear(image_dim, num_classes).to(device)

    # ────────────────────────────────
    # Dataset & DataLoader
    # ────────────────────────────────
    # No transform: ImageFolder will return PIL images
    full_ds = ImageFolder(DATA_PATH)
    train_size = int(0.9 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # ────────────────────────────────
    # Optimizer & Loss
    # ────────────────────────────────
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ────────────────────────────────
    # Training Loop
    # ────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=80)
        total_loss = 0.0

        for images, labels in loop:
            # images: List[PIL.Image]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = clip.get_image_features(**inputs)  # (B, image_dim)
            logits = classifier(feats)                   # (B, num_classes)
            loss   = criterion(logits, labels.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n if loop.n else 1))

        # ────────────────────────────────
        # Validation
        # ────────────────────────────────
        classifier.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for images, labels in val_loader:
                inputs = processor(images=images, return_tensors="pt").to(device)
                feats  = clip.get_image_features(**inputs)
                preds  = classifier(feats).argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total   += labels.size(0)
        acc = correct / total * 100
        print(f"Validation Accuracy after epoch {epoch}: {acc:.2f}%")

    # ────────────────────────────────
    # Save models
    # ────────────────────────────────
    torch.save(classifier.state_dict(), os.path.join(SAVE_DIR, "classifier_head.pth"))
    clip.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)

    print("✅ Fine‑tuning complete.")
    print(f"Models saved under `{SAVE_DIR}/`")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
