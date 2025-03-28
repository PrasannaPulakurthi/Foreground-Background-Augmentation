import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# -------------------------------
# 1. Load and prepare the image
# -------------------------------
image_path = 'person.jpg'  # Replace with your image file path
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()
# Convert BGR to RGB (required for both MediaPipe and SAM2)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Resize for consistency (adjust as needed)
image_rgb = cv2.resize(image_rgb, (256, 256))
h, w, _ = image_rgb.shape

# -------------------------------
# 2. Generate a rough segmentation mask with MediaPipe
# -------------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    results = segmenter.process(image_rgb)
    # Create a binary mask using a threshold (tweak threshold as needed)
    mediapipe_mask = results.segmentation_mask > 0.5

# Convert boolean mask to float32 (values 0.0 and 1.0)
mediapipe_mask = mediapipe_mask.astype(np.float32)
# Expand dims to get shape (1, H, W) as expected by SAM2
mask_input = np.expand_dims(mediapipe_mask, axis=0)

print("mask_input shape:", mask_input.shape)
print("Image shape:", image_rgb.shape)

# -------------------------------
# 3. Define point prompts: a foreground (center) and background (diagonal corners)
# -------------------------------
# Foreground: Middle of the image (or you could use a bounding box center if available)
fg_point = np.array([[w // 2, h // 2]], dtype=np.float32)

# Background: Diagonal corner points
bg_points = np.array([
    [0, 0],         # top-left
    [w - 1, 0],     # top-right
    [0, h - 1],     # bottom-left
    [w - 1, h - 1]  # bottom-right
], dtype=np.float32)

# Combine points: foreground points get label 1, background points label 0
point_coords = np.concatenate([fg_point, bg_points], axis=0)
point_labels = np.concatenate([np.ones(len(fg_point), dtype=np.int32),
                               np.zeros(len(bg_points), dtype=np.int32)], axis=0)

print("Point coords:", point_coords)
print("Point labels:", point_labels)

# -------------------------------
# 4. Initialize SAM2 predictor and set image
# -------------------------------
checkpoint = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
predictor.set_image(image_rgb)

# -------------------------------
# 5. Run SAM2 prediction using both the mask prompt and point prompts
# -------------------------------
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks, _, _ = predictor.predict(
         mask_input=mask_input,       # Rough mask prompt from MediaPipe
         point_coords=point_coords,     # Combined point coordinates
         point_labels=point_labels,     # Corresponding point labels (1: foreground, 0: background)
         multimask_output=False         # Use a single best mask output
    )
sam_mask = masks[0]  # The refined mask from SAM2

# -------------------------------
# 6. Visualize the results
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Show original image with overlayed MediaPipe rough mask
axes[0].imshow(image_rgb)
axes[0].imshow(mediapipe_mask, alpha=0.5, cmap="jet")
axes[0].set_title("MediaPipe Rough Mask")
axes[0].axis("off")

# Show the image with the point prompts overlayed
axes[1].imshow(image_rgb)
axes[1].scatter(point_coords[:1, 0], point_coords[:1, 1], color='red', marker='x', s=100, label="Foreground")
axes[1].scatter(point_coords[1:, 0], point_coords[1:, 1], color='blue', marker='o', s=100, label="Background")
axes[1].set_title("Point Prompts")
axes[1].legend()
axes[1].axis("off")

# Show SAM2 refined segmentation mask
axes[2].imshow(image_rgb)
axes[2].imshow(sam_mask, alpha=0.5, cmap="jet")
axes[2].set_title("SAM2 Refined Mask")
axes[2].axis("off")

plt.show()
