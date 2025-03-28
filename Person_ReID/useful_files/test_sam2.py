import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def fallback_center_grid(w, h, grid_size_x=3, grid_size_y=7, spacing=10):
    """
    Creates a grid of points around the image center,
    with the specified grid dimensions and spacing.
    """
    center_x, center_y = w // 2, h // 2
    fg_points = []
    
    # For a 3×7 grid, i will run over -1..1 and j over -3..3 (if grid_size_y=7)
    # Adjust the offset logic to suit your preference.
    half_x = grid_size_x // 2
    half_y = grid_size_y // 2
    
    for i in range(-half_x, half_x + 1):
        for j in range(-half_y, half_y + 1):
            point_x = center_x + i * spacing
            point_y = center_y + j * spacing
            # Check bounds
            if 0 <= point_x < w and 0 <= point_y < h:
                fg_points.append([point_x, point_y])
    
    return np.array(fg_points)


# Define paths to the checkpoint and configuration files
checkpoint = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

# Initialize the SAM2 predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Read the image using OpenCV
image_path = 'person.jpg'
image_bgr = cv2.imread(image_path)
# Convert BGR to RGB (for both Mediapipe and SAM2)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Get image dimensions
h, w, _ = image_rgb.shape

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# Process the image and detect pose landmarks
results = pose_detector.process(image_rgb)

# Use detected landmarks as foreground points if available;
# otherwise, fall back to the center of the image.
if results.pose_landmarks:
    # Collect valid pose landmarks (within image bounds)
    valid_pose_points = []
    for lm in results.pose_landmarks.landmark:
        px = int(lm.x * w)
        py = int(lm.y * h)
        if 0 <= px < w and 0 <= py < h:
            valid_pose_points.append([px, py])

    if len(valid_pose_points) > 0:
        valid_pose_points = np.array(valid_pose_points)

        # 1. Compute bounding box around the valid pose landmarks
        min_x = valid_pose_points[:, 0].min()
        max_x = valid_pose_points[:, 0].max()
        min_y = valid_pose_points[:, 1].min()
        max_y = valid_pose_points[:, 1].max()

        # Optionally expand the box a bit
        bbox_expand = 10
        min_x = max(min_x - bbox_expand, 0)
        min_y = max(min_y - bbox_expand, 0)
        max_x = min(max_x + bbox_expand, w - 1)
        max_y = min(max_y + bbox_expand, h - 1)

        # 2. Find bounding box center
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # 3. Create a small cluster around the bounding box center
        #    For example, a 3×3 grid with spacing=5
        extra_points = []
        grid_size_x = 3
        grid_size_y = 6
        spacing_x = 5
        spacing_y = 10
        half_x = grid_size_x // 2
        half_y = grid_size_y // 2
        for i in range(-half_x, half_x + 1):
            for j in range(-half_y, half_y + 1):
                px = center_x + i * spacing_x
                py = center_y + j * spacing_y
                if 0 <= px < w and 0 <= py < h:
                    extra_points.append([px, py])
        extra_points = np.array(extra_points)

        # Combine the original pose landmarks with the new center points
        fg_points = np.concatenate([valid_pose_points, extra_points], axis=0)
        fg_points = np.unique(fg_points, axis=0)  # remove duplicates if desired

        print(f"Pose landmarks: {len(valid_pose_points)}; "
            f"Center cluster: {len(extra_points)}; "
            f"Total FG: {len(fg_points)}")

    else:
        # No valid landmarks => fallback strategy (e.g., center grid)
        fg_points = fallback_center_grid(w, h, grid_size_x=3, grid_size_y=7, spacing=10)
        print(f"No valid pose landmarks; using fallback with {len(fg_points)} points.")

else:
    # No pose landmarks at all => fallback
    fg_points = fallback_center_grid(w, h, grid_size_x=3, grid_size_y=7, spacing=10)
    print(f"No pose landmarks detected for {image_path}; "
          f"using fallback grid of {len(fg_points)} points as foreground.")




# Create foreground labels (1 indicates foreground)
fg_labels = np.ones(len(fg_points), dtype=int)

# Define background points at the extreme corners of the image
bg_points = np.array([
    [0, 0],          # top-left corner
    [w - 1, 0],      # top-right corner
    [0, h - 1],      # bottom-left corner
    [w - 1, h - 1]   # bottom-right corner
])
bg_labels = np.zeros(len(bg_points), dtype=int)  # 0 indicates background

# Combine foreground and background points and labels
point_coords = np.concatenate([fg_points, bg_points], axis=0)
point_labels = np.concatenate([fg_labels, bg_labels], axis=0)

# Set the image in the SAM2 predictor (expects a NumPy array in RGB format)
predictor.set_image(image_rgb)

# Run SAM2 prediction with the given point prompts
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False  # Use single best mask output
    )
mask = masks[0]

# Visualize the results using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show the original image with foreground and background points
axes[0].imshow(image_rgb)
axes[0].scatter(fg_points[:, 0], fg_points[:, 1], color='red', marker='x', s=50, label="Foreground (Pose)")
axes[0].scatter(bg_points[:, 0], bg_points[:, 1], color='blue', marker='o', s=50, label="Background")
axes[0].set_title("Original Image with Points")
axes[0].axis("off")
#axes[0].legend()

# Show the segmentation mask overlay
axes[1].imshow(image_rgb)
axes[1].imshow(mask, alpha=0.5, cmap="jet")
axes[1].set_title("Segmentation Mask")
axes[1].axis("off")

plt.show()
