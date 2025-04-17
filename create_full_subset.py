import os
import shutil
import random

# CONFIGURATION
source_dir = r"C:\Users\Om\COMP9517\PROJECT\Image-Classification-CV\archive\Aerial_Landscapes"
target_dir = "subset"
train_ratio = 0.8  # 80% train, 20% test
random.seed(42)

def create_subset_all_classes():
    all_classes = [cls for cls in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cls))]
    
    for phase in ['train', 'test']:
        for cls in all_classes:
            os.makedirs(os.path.join(target_dir, phase, cls), exist_ok=True)

    for cls in all_classes:
        class_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)

        num_total = len(images)
        num_train = int(train_ratio * num_total)
        train_imgs = images[:num_train]
        test_imgs = images[num_train:]

        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'train', cls, img)
            shutil.copy(src, dst)

        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'test', cls, img)
            shutil.copy(src, dst)

        print(f"✅ {cls}: {len(train_imgs)} train, {len(test_imgs)} test")

    print(f"\n📦 Full dataset subset created at: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    create_subset_all_classes()
