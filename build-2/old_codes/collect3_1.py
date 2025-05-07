import os
import cv2
import dlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

# Define paths
source_dir = "D:/pro_dis/original_dataset_even_frames"
output_dir = "/collected_data"

# Load the detector and predictor (dlib models should be loaded outside of the multiprocessing function to avoid reloading)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/PycharmProjects/pro_dis_2/model/face_weights.dat")

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

        # Calculate padding to fit the target dimensions
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = max(width_diff // 2, 0)
        pad_right = max(width_diff - pad_left, 0)
        pad_top = max(height_diff // 2, 0)
        pad_bottom = max(height_diff - pad_top, 0)

        # Adjust padding to ensure it doesn’t exceed image boundaries
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Crop and resize the mouth region
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        # Save the cropped mouth region to the output directory
        output_frame_path = os.path.join(output_word_path, frame_file)
        cv2.imwrite(output_frame_path, lip_frame)

        print(f"Cropped and saved frame: {output_frame_path}")
        return  # Exit after processing the first detected face


def process_word_folder(word_folder):
    word_path = os.path.join(source_dir, word_folder)

    # Create output directory for cropped frames
    output_word_path = os.path.join(output_dir, word_folder)
    os.makedirs(output_word_path, exist_ok=True)

    frame_files = os.listdir(word_path)

    # Use ProcessPoolExecutor to process frames concurrently
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_file, word_path, output_word_path) for frame_file in
                   frame_files]
        for future in futures:
            future.result()  # Wait for all futures to complete


if __name__ == "__main__":
    # Process each word folder sequentially
    word_folders = os.listdir(source_dir)
    for word_folder in word_folders:
        print(f"Processing word folder: {word_folder}")
        process_word_folder(word_folder)

    print("Mouth cropping completed for all frames.")
