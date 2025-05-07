import os
import cv2
import json
import numpy as np
import imageio.v2 as imageio

# Path to the folder containing word folders with frames
word_folders_path = r'C:\Users\avikd\Videos\project_videos\original_dataset_processed_frames\set_01'

# List to store all frames for each word sequence
all_words = []
labels = []

# Set FPS for video creation
fps = 30  # You can adjust this value as needed

# Iterate over each word folder
for word_folder in sorted(os.listdir(word_folders_path)):
    word_path = os.path.join(word_folders_path, word_folder)
    print(f'Processing: "{word_folder}"')

    # Check if the path is a directory (word folder)
    if os.path.isdir(word_path):
        word_frames = []

        # Iterate over each frame image in sequential order
        frame_files = sorted(
            [f for f in os.listdir(word_path) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        for frame_file in frame_files:
            frame_path = os.path.join(word_path, frame_file)

            # Read the frame image
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Convert frame to a list of pixel values (height × width × channels)
                frame_data = frame.tolist()
                word_frames.append(frame_data)

        # Add the word frames to the all_words list and create data.txt and video.mp4
        if word_frames:
            all_words.append(word_frames)
            labels.append(word_folder)

            # Save word frames to a data.txt file in JSON format
            txt_path = os.path.join(word_path, "data.txt")
            with open(txt_path, "w") as f:
                f.write(json.dumps(word_frames))

            # Create video.mp4 using the frames in all_words without saving .png files
            video_path = os.path.join(word_path, "video.mp4")
            images = []

            for frame_data in word_frames:
                # Convert the frame data back to an array and add to the images list
                img_array = np.array(frame_data, dtype=np.uint8)
                images.append(img_array)

            # Save the video from the list of images
            imageio.mimsave(video_path, images, fps=fps)

            # Clear the all_words list for the next iteration
            all_words.clear()

print("\nData collection completed.")
print("The data.txt and video.mp4 files have been created in their corresponding word folders.")
