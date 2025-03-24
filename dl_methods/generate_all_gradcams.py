import os
import torch
from datetime import datetime
from dl_methods.gradcam import generate_gradcam
from torchvision import models
import torch.nn as nn

def run_batch_gradcam(model_path="models/resnet18_finetuned.pth"):
    print("\n📸 Generating Grad-CAM for 1 image per class...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet-18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 15)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dir = "data/test"
    class_names = sorted(os.listdir(test_dir))

    # Set up output dir with timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = f"gradcam_outputs/ResNet18_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "report.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Grad-CAM Report\nModel: ResNet-18\nTimestamp: {timestamp}\n\n")

        for cls in class_names:
            class_dir = os.path.join(test_dir, cls)
            if not os.path.isdir(class_dir): continue

            images = os.listdir(class_dir)
            if len(images) == 0: continue

            image_path = os.path.join(class_dir, images[0])
            pred_class = generate_gradcam(model, image_path, class_names, output_dir)

            log_file.write(f"{os.path.basename(image_path)} → Predicted: {pred_class}\n")

    print(f"\n✅ Grad-CAM report saved to: {log_path}")
