# filename: clip_infer_save.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import random
import warnings
from datetime import datetime

# 🚫 Suppress HuggingFace cache warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "clip_model"
DATA_PATH = "archive/Aerial_Landscapes"
RESULT_DIR = "results/LLM_results"
SUMMARY_PATH = "results/summary_report.txt"

model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

categories = sorted(os.listdir(DATA_PATH))

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
save_path = os.path.join(RESULT_DIR, f"llm_eval_{timestamp}")
os.makedirs(save_path, exist_ok=True)

report_lines = []
correct = 0
total = 0

print("🧪 Running inference on 10 images per category...")

for category in categories:
    cat_dir = os.path.join(DATA_PATH, category)
    images = [img for img in os.listdir(cat_dir) if img.lower().endswith((".jpg", ".png"))]
    sample_images = random.sample(images, min(100, len(images)))

    report_lines.append(f"Category: {category}")
    for img in sample_images:
        path = os.path.join(cat_dir, img)
        image = Image.open(path).convert("RGB")
        inputs = processor(text=categories, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        pred_label = categories[probs.argmax().item()]
        is_correct = "✅" if pred_label == category else "❌"
        report_lines.append(f" - {img} → Predicted: {pred_label} {is_correct}")
        total += 1
        if pred_label == category:
            correct += 1
    report_lines.append("")

# Calculate overall accuracy
accuracy = (correct / total) * 100
report_lines.insert(0, f"🧠 CLIP LLM Accuracy: {accuracy:.2f}%")
report_lines.insert(1, f"Evaluation Timestamp: {timestamp}")
report_lines.insert(2, f"Total images: {total}")
report_lines.insert(3, "-" * 40)

# Save detailed report
report_file = os.path.join(save_path, "report.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# Append to summary report
with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n----- CLIP LLM Evaluation ({timestamp}) -----\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")

print("\n📄 Report saved to:", report_file)
print(f"📌 Summary appended to: {SUMMARY_PATH}")
