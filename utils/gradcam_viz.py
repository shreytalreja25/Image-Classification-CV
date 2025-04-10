# utils/gradcam_viz.py
import os
import matplotlib.pyplot as plt
import cv2

def save_gradcam_visual(original_img, heatmap_img, true_class, pred_class, output_path, filename):
    os.makedirs(output_path, exist_ok=True)

    # Convert BGR (OpenCV) to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    # Stitch side by side
    stitched = cv2.hconcat([original_rgb, heatmap_rgb])

    # Plot and title
    plt.figure(figsize=(10, 5))
    plt.imshow(stitched)
    title = f"GT: {true_class}   |   Pred: {pred_class}"
    color = "green" if true_class == pred_class else "red"
    plt.title(title, fontsize=14, color=color)
    plt.axis('off')

    # Save image
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"🖼️ Saved visual → {save_path}")
