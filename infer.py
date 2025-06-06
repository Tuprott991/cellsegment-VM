import os
import random
import numpy as np
import torch
from model_arch import InstanSegModel
import matplotlib.pyplot as plt

def show_npy_image(image, mask=None, title=""):
    plt.figure(figsize=(8,4) if mask is not None else (4,4))
    if mask is not None:
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap='jet')  # chỉ hiển thị mask
        plt.title("Predicted Mask")
        plt.axis("off")
    else:
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Inference ---
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InstanSegModel(in_ch=3, base_ch=32, Dp=2, De=4).to(device)
model.load_state_dict(torch.load("model_epoch_17.pth", map_location=device))
model.eval()

# Pick a random image
images_dir = "prepared/images"
image_files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
img_name = random.choice(image_files)
img_path = os.path.join(images_dir, img_name)
image = np.load(img_path)  # shape: (256,256,3)
image_tensor = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0) / 255.0  # [1,3,256,256]
image_tensor = image_tensor.to(device)

print(f"Processing image: {img_name}")

# Inference
with torch.no_grad():
    S, P, E, O = model(image_tensor)

# Segment instances
instances = model.segment_instances(S, P, E, O, iou_threshold=0.6)

print(f"Predicted {len(instances)} instances for image {img_name}")

# Use mathplotlib and PIL to visualize all instances
# for idx, mask in enumerate(instances):
#     mask_np = mask.cpu().numpy()  # Convert to numpy array
#     show_npy_image(image, mask_np, title=f"Instance {idx+1} Mask for {img_name}")

# Show S
show_npy_image(image, S[0, 0].cpu().numpy(), title=f"Seed map for {img_name}")