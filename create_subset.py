import os
import shutil
import random

# CONFIGURATION
source_dir = r"C:\Users\Om\COMP9517\PROJECT\Image-Classification-CV\archive\Aerial_Landscapes"
target_dir = "subset"
selected_classes = ["Airport", "Forest", "Beach"]  # <-- Pick any 3-5
num_train = 640
num_test = 160
random.seed(42)

def create_subset():
    for phase in ['train', 'test']:
        for cls in selected_classes:
            os.makedirs(os.path.join(target_dir, phase, cls), exist_ok=True)

    for cls in selected_classes:
        class_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)
        
        train_imgs = images[:num_train]
        test_imgs = images[num_train:num_train + num_test]

        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'train', cls, img)
            shutil.copy(src, dst)

        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'test', cls, img)
            shutil.copy(src, dst)

    print(f"✅ Subset created at: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    create_subset()
