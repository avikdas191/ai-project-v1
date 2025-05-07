import os
import cv2
import numpy as np

# Define paths
source_dir = "/collected_data"  # Path to the cropped frames
output_file = "folders_with_detection_issues.txt"  # Output file to save folders with issues

# Set threshold for difference (can be tuned based on test results)
DIFFERENCE_THRESHOLD = 10000  # Adjust this based on testing


# Function to check if an image has lips detected correctly
def is_lip_detected(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to determine if the frame has distinct edges
    edges = cv2.Canny(gray_frame, 50, 150)
    edge_score = np.sum(edges)  # Sum of edge intensities

    # If edge score is low, it means the frame lacks distinct features (like the lip region)
    return edge_score > DIFFERENCE_THRESHOLD


# List to store folders with detection issues
folders_with_issues = []

# Iterate over each word folder
for word_folder in os.listdir(source_dir):
    word_path = os.path.join(source_dir, word_folder)

    # Skip if not a directory
    if not os.path.isdir(word_path):
        continue

    issue_found = False  # Flag to mark if issue is found in the folder

    # Check each frame in the folder
    for frame_file in os.listdir(word_path):
        frame_path = os.path.join(word_path, frame_file)
        frame = cv2.imread(frame_path)

        # Check if the frame has a valid lip detection
        if not is_lip_detected(frame):
            print(f"Issue detected in {frame_file} of folder {word_folder}")
            folders_with_issues.append(word_folder)
            issue_found = True
            break  # Stop further checks if an issue is found in the folder

# Save folders with issues to a text file
with open(output_file, "w") as f:
    for folder in folders_with_issues:
        f.write(folder + "\n")

print(f"Folders with detection issues saved to {output_file}")
