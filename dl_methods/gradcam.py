import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from PIL import Image

def generate_gradcam(model, image_path, class_names, output_path, model_name="resnet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # === Register target layer ===
    if model_name == "resnet":
        target_layer = model.layer4[1].conv2
    elif model_name == "efficientnet":
        target_layer = model._blocks[-1]._project_conv
    elif model_name == "mobilenet":
        target_layer = model.features[-1][0]
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # === Load & preprocess ===
    img = Image.open(image_path).convert("RGB")
    original_np = np.array(img)
    original_cv = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # === Forward pass ===
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # === Backward ===
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # === Grad-CAM ===
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    cam = cv2.resize(cam, original_cv.shape[:2][::-1])

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_cv, 0.5, heatmap, 0.5, 0)

    # === Combine original + overlay ===
    stitched = np.hstack((original_cv, overlay))

    # === Save output ===
    os.makedirs(output_path, exist_ok=True)
    stitched_path = os.path.join(output_path, f"stitched_gradcam_{model_name}_{os.path.basename(image_path)}")
    cv2.imwrite(stitched_path, stitched)

    # === Get GT and prediction
    true_class = os.path.basename(os.path.dirname(image_path))
    predicted_class = class_names[pred_class]

    print(f"✅ Saved: {stitched_path} | GT: {true_class}, Pred: {predicted_class}")
    return true_class, predicted_class
