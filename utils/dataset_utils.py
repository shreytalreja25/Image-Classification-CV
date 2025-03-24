import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_dir, output_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    categories = [cat for cat in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cat))]
    summary = {}

    print(f"\n📁 Splitting dataset from: {source_dir}")
    print(f"💾 Output directory: {output_dir}\n")

    for category in categories:
        category_path = os.path.join(source_dir, category)
        images = os.listdir(category_path)
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_images = images[:split_idx]
        test_images = images[split_idx:]
        summary[category] = {
            'train': len(train_images),
            'test': len(test_images)
        }

        print(f"\n🔹 Splitting category: {category}")
        for split_name, split_images in [('train', train_images), ('test', test_images)]:
            split_path = Path(output_dir) / split_name / category
            split_path.mkdir(parents=True, exist_ok=True)

            for img in tqdm(split_images, desc=f"   ➤ {split_name.upper()} [{category}]", ncols=80):
                src = Path(category_path) / img
                dst = split_path / img
                try:
                    shutil.copyfile(src, dst)
                except Exception as e:
                    print(f"⚠️ Could not copy {img}: {e}")

    # Final Summary
    print("\n📊 Split Summary:")
    total_train, total_test = 0, 0
    for category, stats in summary.items():
        print(f"   {category}: {stats['train']} train / {stats['test']} test")
        total_train += stats['train']
        total_test += stats['test']

    print("\n✅ All categories processed successfully!")
    print(f"🗂️  Total training images: {total_train}")
    print(f"🧪 Total testing images:  {total_test}")
    print(f"\n📍 Check your data at: {output_dir}/train/ and {output_dir}/test/\n")
