import os
import cv2
import dlib
from concurrent.futures import ProcessPoolExecutor

# Define paths
source_dir = r"C:\Users\avikd\Videos\project_videos\original_dataset_even_frames"
output_dir = r"C:\Users\avikd\Videos\project_videos\original_dataset_DLib_cropped_frames"

# Load the detector and predictor (dlib models should load outside the multiprocessing function to avoid reloading)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/PycharmProjects/source project files/models/shape_predictor_68_face_landmarks.dat")

# Mouth crop dimensions
LIP_HEIGHT = 80
LIP_WIDTH = 112


def process_frame(frame_file, word_path, output_word_path):
    frame_path = os.path.join(word_path, frame_file)

    # Load the frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: Could not read frame {frame_path}. Skipping.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Only process if a face is detected
    for face in faces:
        landmarks = predictor(gray, face)

        # Get mouth coordinates
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Define an additional buffer around the lips
        lip_left_buffer = int(0.25 * LIP_WIDTH)  # Add 25% of LIP_WIDTH as a buffer
        lip_right_buffer = int(0.25 * LIP_WIDTH)  # Add 25% of LIP_WIDTH as a buffer
        lip_top_buffer = int(0.2 * LIP_HEIGHT)  # Add 20% of LIP_HEIGHT as a buffer
        lip_bottom_buffer = int(0.35 * LIP_HEIGHT)  # Add 35% of LIP_HEIGHT as a buffer

        # Calculate padding to fit the target dimensions
        width_diff = LIP_WIDTH - (lip_right - lip_left) + lip_left_buffer + lip_right_buffer
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top) + lip_top_buffer + lip_bottom_buffer
        pad_left = max(width_diff // 2, 0)
        pad_right = max(width_diff - pad_left, 0)
        pad_top = max(height_diff // 2, 0)
        pad_bottom = max(height_diff - pad_top, 0)

        # Adjust padding to ensure it doesn’t exceed image boundaries
        pad_left = min(pad_left + lip_left_buffer, lip_left)
        pad_right = min(pad_right + lip_right_buffer, frame.shape[1] - lip_right)
        pad_top = min(pad_top + lip_top_buffer, lip_top)
        pad_bottom = min(pad_bottom + lip_bottom_buffer, frame.shape[0] - lip_bottom)

        # Crop and resize the mouth region
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        # Save the cropped mouth region to the output directory
        output_frame_path = os.path.join(output_word_path, frame_file)
        cv2.imwrite(output_frame_path, lip_frame)

        print(f"Cropped and saved frame: {output_frame_path}")
        return  # Exit after processing the first detected face


def process_word_folder(set_folder, word_folder):
    word_path = os.path.join(set_folder, word_folder)

    # Create output directory for cropped frames
    output_word_path = os.path.join(output_dir, os.path.basename(set_folder), word_folder)
    os.makedirs(output_word_path, exist_ok=True)

    frame_files = os.listdir(word_path)

    # Use ProcessPoolExecutor to process frames concurrently
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_file, word_path, output_word_path) for frame_file in
                   frame_files]
        for future in futures:
            future.result()  # Wait for all futures to complete


if __name__ == "__main__":
    # Iterate over each set folder and process each word folder within
    set_folders = os.listdir(source_dir)
    for set_folder in set_folders:
        full_set_path = os.path.join(source_dir, set_folder)
        if os.path.isdir(full_set_path):
            word_folders = os.listdir(full_set_path)
            for word_folder in word_folders:
                print(f"Processing word folder: '{word_folder}' in: '{set_folder}'")
                process_word_folder(full_set_path, word_folder)

    print("Mouth cropping completed for all frames.")