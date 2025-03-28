import cv2
import mediapipe as mp
import numpy as np

# Path to the input image file.
image_path = 'person1.jpg'  # Replace with your image file path

# Read the image using OpenCV.
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Resize the image to (width, height) = (128, 256).
# image = cv2.resize(image, (128, 256))

# Convert the resized image from BGR to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Selfie Segmentation.
mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    # Process the RGB image to obtain the segmentation mask.
    results = segmenter.process(image_rgb)

    # Create a binary mask using a threshold.
    mask = results.segmentation_mask > 0.1  # Adjust this threshold as needed.

    # For visualization, create a blurred version of the resized image.
    blurred_image = cv2.GaussianBlur(image, (55, 55), 0)

    # Composite the images: keep the resized image where the mask is true (human)
    # and use the blurred image elsewhere.
    output_image = np.where(mask[..., None], image, blurred_image)

# Display the resulting image.
cv2.imshow('Human Segmentation Mask', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
