import os
import cv2
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import imageio.v2 as imageio

# Function to process a single frame
def process_frame(frame_file, word_folder_path):
    frame_path = os.path.join(word_folder_path, frame_file)
    frame = cv2.imread(frame_path)
    return frame if frame is not None else None  # Keep frame in original format

# Function to save all processed data for a word folder
def save_all_data(word_folder_path, frames):
    # Convert frames to lists for saving in data.txt
    frame_data = [frame.tolist() for frame in frames]  # Convert frames to lists only here
    data_path = os.path.join(word_folder_path, 'data.txt')
    with open(data_path, 'w') as f:
        json.dump(frame_data, f)
    print("Saved data.txt")

    # Save video.mp4 using imageio.mimsave without modifying data type
    video_path = os.path.join(word_folder_path, 'video.mp4')
    imageio.mimsave(video_path, frames, fps=25)  # Use frames as-is
    print("Saved video.mp4")

# Function to process all frames in a word folder
def process_word_folder(word_folder_path):
    frame_files = sorted(os.listdir(word_folder_path))
    frames = []

    # Use ProcessPoolExecutor to process frames concurrently within the folder using submit
    with ProcessPoolExecutor() as executor:
        future_to_frame = {executor.submit(process_frame, frame_file, word_folder_path): frame_file for frame_file in frame_files}

        # Collect the processed frames as each future completes
        for future in as_completed(future_to_frame):
            frame = future.result()
            if frame is not None:
                frames.append(frame)

    # Save the frames data as data.txt and video.mp4
    if frames:
        save_all_data(word_folder_path, frames)
    print(f"Completed processing for folder: {word_folder_path}")

# Main function to process all word folders
def main():
    root_folder = "D:/PycharmProjects/pro_dis_2/collected_data/processed_cropped"
    word_folders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]

    for word_folder in word_folders:
        print(f"Processing folder: {word_folder}")
        process_word_folder(word_folder)

if __name__ == "__main__":
    main()
