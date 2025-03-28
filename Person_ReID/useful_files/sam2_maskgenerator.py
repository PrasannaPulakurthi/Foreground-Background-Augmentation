import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

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



def save_output(image_path, pred, prediction_dir, root_dir):
    # Squeeze the predicted mask (assumed to be a numpy array)
    predict_np = pred.squeeze()

    # Convert normalized prediction to an RGB image
    imo = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('RGB')

    # Use cv2 to read the original image and get its dimensions
    orig_img = cv2.imread(image_path)
    h, w, _ = orig_img.shape
    imo = imo.resize((w, h), resample=Image.BILINEAR)

    # Derive the output filename and folder structure
    img_name = os.path.basename(image_path)
    out_filename = img_name
    rel_path = os.path.relpath(image_path, root_dir)
    rel_folder = os.path.dirname(rel_path)

    # Create the corresponding folder under prediction_dir
    out_dir = os.path.join(prediction_dir, rel_folder)
    os.makedirs(out_dir, exist_ok=True)

    # Save the mask image
    out_path = os.path.join(out_dir, out_filename)
    imo.save(out_path)


def get_image_paths(root_dir, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(exts):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    return image_paths


# Define paths to the SAM2 model checkpoint and configuration file
checkpoint = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

# Initialize the SAM2 predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Initialize Mediapipe Pose for human landmark detection
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define the input folder (root of images) and the target folder for saving masks
# input_dir = "data/Market/pytorch/train/"
# prediction_dir = "data/Market/pytorch/train_mask/"
input_dir = "data/DukeMTMC-reID/pytorch/train/"
prediction_dir = "data/DukeMTMC-reID/pytorch/train_mask/"


# Gather all image paths from the input directory
image_paths = get_image_paths(input_dir)
print(f"Found {len(image_paths)} images in {input_dir}")

# Process each image in the input folder
for image_path in image_paths:
    # Read image using cv2 (returns BGR)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to read {image_path}")
        continue

    # Convert BGR to RGB for both Mediapipe and SAM2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Get image dimensions
    h, w, _ = image_rgb.shape

    # Use Mediapipe Pose to detect human landmarks
    results = pose_detector.process(image_rgb)

    ##########################################################    
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

    # Combine the foreground and background points and labels
    point_coords = np.concatenate([fg_points, bg_points], axis=0)
    point_labels = np.concatenate([fg_labels, bg_labels], axis=0)

    # Perform segmentation prediction using SAM2
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False  # Use the best mask only
        )
    mask = masks[0]

    # Save the mask output to the target folder, preserving folder structure
    save_output(image_path, mask, prediction_dir, input_dir)
    print(f"Processed and saved mask for: {image_path}")

print("All images processed.")
