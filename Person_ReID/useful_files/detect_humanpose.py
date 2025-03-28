import cv2
import mediapipe as mp
import numpy as np


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



# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Read the image
image_path = 'person.jpg'
image = cv2.imread(image_path)
# Convert the BGR image to RGB before processing.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and detect pose landmarks
results = pose.process(image_rgb)

if results.pose_landmarks:
    h, w, _ = image.shape
    # Iterate through each landmark and print its coordinates.
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # Convert normalized coordinates to pixel values
        x, y = int(landmark.x * w), int(landmark.y * h)
        print(f"Landmark {idx}: ({x}, {y})")
    
    # Optionally, draw the landmarks on the image for visualization.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Show the annotated image
    cv2.imshow('Pose Estimation', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose landmarks detected.")
