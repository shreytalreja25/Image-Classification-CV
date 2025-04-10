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

    # === Register correct target layer based on model ===
    if model_name == "resnet":
        target_layer = model.layer4[1].conv2
    elif model_name == "efficientnet":
        target_layer = model._blocks[-1]._project_conv
    elif model_name == "mobilenet":
        target_layer = model.features[-1][0]
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # Use full backward hook to avoid deprecation warning
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # === Load and preprocess image ===
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # === Forward pass ===
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # === Backward pass ===
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # === Get Grad-CAM map ===
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, img.size)
    cam -= np.min(cam)
    cam /= np.max(cam)

    # === Overlay heatmap ===
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_img = np.array(img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"gradcam_{model_name}_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, overlay)

    print(f"✅ Saved Grad-CAM: {os.path.basename(image_path)} → predicted: {class_names[pred_class]}")
    return class_names[pred_class]


# def generate_gradcam(model, image_path, class_names, output_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     gradients = []
#     activations = []

#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     # Hook into last conv layer
#     target_layer = model.layer4[1].conv2
#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_backward_hook(backward_hook)

#     # Load + preprocess image
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_rgb)

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     input_tensor = transform(img_pil).unsqueeze(0).to(device)

#     # Forward pass
#     output = model(input_tensor)
#     pred_class = output.argmax(dim=1).item()

#     # Backward pass
#     model.zero_grad()
#     class_score = output[0, pred_class]
#     class_score.backward()

#     grad = gradients[0].cpu().data.numpy()[0]
#     act = activations[0].cpu().data.numpy()[0]

#     # Grad-CAM calculation
#     weights = np.mean(grad, axis=(1, 2))
#     cam = np.zeros(act.shape[1:], dtype=np.float32)

#     for i, w in enumerate(weights):
#         cam += w * act[i]

#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
#     cam -= np.min(cam)
#     cam /= np.max(cam)

#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

#     os.makedirs(output_path, exist_ok=True)
#     out_path = os.path.join(output_path, f"gradcam_{os.path.basename(image_path)}")
#     cv2.imwrite(out_path, overlay)

#     print(f"✅ Saved Grad-CAM: {os.path.basename(image_path)} → {class_names[pred_class]}")
#     return class_names[pred_class]
