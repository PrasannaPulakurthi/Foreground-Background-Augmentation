import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
    rel_path = os.path.relpath(image_path, root_dir)
    rel_folder = os.path.dirname(rel_path)

    # Create the corresponding folder under prediction_dir
    out_dir = os.path.join(prediction_dir, rel_folder)
    os.makedirs(out_dir, exist_ok=True)

    # Save the mask image
    out_path = os.path.join(out_dir, img_name)
    imo.save(out_path)


def get_image_paths(root_dir, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(exts):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    return image_paths


# -------------------------------------------------------------------
# Setup: Initialize SAM2 predictor and MediaPipe Selfie Segmentation
# -------------------------------------------------------------------
checkpoint = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Initialize MediaPipe Selfie Segmentation (for a rough human mask)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Define input and output directories
# dataset = "Market" 
dataset = "DukeMTMC-reID"

input_dir = "data/"+dataset+"/pytorch/train/"
prediction_dir = "data/"+dataset+"/pytorch/train_mask/"

# Gather all image paths from the input directory
image_paths = get_image_paths(input_dir)
print(f"Found {len(image_paths)} images in {input_dir}")

# -------------------------------------------------------------------
# Process each image: generate rough mask, add point prompts, refine with SAM2, and save output
# -------------------------------------------------------------------
# For consistency, we resize images to a fixed resolution (e.g., 256x256) before feeding to SAM2.
# Later, the output mask is resized back to the original resolution.
RESIZE_DIM = (256, 256)

for image_path in image_paths:
    # Read image (BGR) and convert to RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to read {image_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize image for SAM2 processing
    image_resized = cv2.resize(image_rgb, RESIZE_DIM)
    predictor.set_image(image_resized)

    # Generate a rough segmentation mask using MediaPipe Selfie Segmentation
    results = segmenter.process(image_resized)
    # Create a binary mask; adjust threshold as needed
    mediapipe_mask = results.segmentation_mask > 0.5
    mediapipe_mask = mediapipe_mask.astype(np.float32)

    # Expand dims to get shape (1, H, W) as expected by SAM2
    mask_input = np.expand_dims(mediapipe_mask, axis=0)

    # Define additional point prompts
    # Use the center of the resized image as the foreground point.
    h_resized, w_resized, _ = image_resized.shape
    fg_point = np.array([[w_resized // 2, h_resized // 2]], dtype=np.float32)

    # Use the diagonal corners as background points.
    bg_points = np.array([
        [0, 0],                  # top-left
        [w_resized - 1, 0],        # top-right
        [0, h_resized - 1],        # bottom-left
        [w_resized - 1, h_resized - 1]  # bottom-right
    ], dtype=np.float32)

    # Combine foreground and background points and define labels (1: foreground, 0: background)
    point_coords = np.concatenate([fg_point, bg_points], axis=0)
    point_labels = np.concatenate([np.ones(len(fg_point), dtype=np.int32),
                                   np.zeros(len(bg_points), dtype=np.int32)], axis=0)

    # print(f"Processing {image_path}")
    # print("Resized image shape:", image_resized.shape,
    #       "mask_input shape:", mask_input.shape,
    #       "point_coords:", point_coords,
    #       "point_labels:", point_labels)

    # Run SAM2 prediction using both the mask prompt and point prompts
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(
            mask_input=mask_input,      # Rough mask prompt from MediaPipe
            point_coords=point_coords,  # Combined point coordinates (foreground and background)
            point_labels=point_labels,  # Corresponding point labels
            multimask_output=False      # Use the single best mask output
        )
    sam_mask = masks[0]  # Refined segmentation mask from SAM2

    # Save the SAM2 refined mask (it will be resized back to the original image resolution)
    save_output(image_path, sam_mask, prediction_dir, input_dir)
    print(f"Processed and saved mask for: {image_path}")

print("All images processed.")

# Clean up MediaPipe resources
segmenter.close()
