import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# Function to apply the processing flow on a cropped image
def process_cropped_image(cropped_image):
    # Step 1: Convert RGB to LAB
    lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)

    # Step 2: Apply CLAHE on the L channel
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
    l_channel_eq = clahe.apply(l_channel)

    # Merge the adjusted L channel back with the original A and B channels
    lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))

    # Step 3: Convert LAB back to BGR
    bgr_image_eq = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)

    # Step 4: Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(bgr_image_eq, (7, 7), 0)

    # Step 5: Apply Bilateral Filter for edge-preserving smoothing
    edge_preserved_image = cv2.bilateralFilter(blurred_image, d=5, sigmaColor=75, sigmaSpace=75)

    # Step 6: Apply Sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(edge_preserved_image, -1, sharpening_kernel)

    # Step 7: Additional Gaussian Blur to reduce harshness from sharpening
    final_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)

    return final_image


# Function to process and save a single frame
def process_and_save_frame(frame_file, word_path, output_word_path):
    frame_path = os.path.join(word_path, frame_file)
    output_frame_path = os.path.join(output_word_path, frame_file)

    # Load the cropped image
    cropped_image = cv2.imread(frame_path)

    # Apply processing
    processed_image = process_cropped_image(cropped_image)

    # Save the processed image to the output directory
    cv2.imwrite(output_frame_path, processed_image)
    print(f"Processed and saved frame: {output_frame_path}")


# Function to process an entire word folder
def process_word_folder(word_folder, set_path, output_set_path):
    word_path = os.path.join(set_path, word_folder)
    output_word_path = os.path.join(output_set_path, word_folder)
    os.makedirs(output_word_path, exist_ok=True)

    # Get all frames in the word folder
    frame_files = os.listdir(word_path)

    # Use ProcessPoolExecutor to process frames within the current word folder
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_and_save_frame, frame_file, word_path, output_word_path) for frame_file in frame_files]

        # Ensure all futures complete
        for future in futures:
            future.result()

    print(f"Completed processing for folder: {word_folder}")


# Main processing setup
def main():
    source_dir = "D:/PycharmProjects/pro_dis_2/collected_data/cropped_datasets_four"
    output_dir = "D:/PycharmProjects/pro_dis_2/collected_data/processed_datasets_five"

    # Iterate over each set folder in the source directory
    set_folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]

    for set_folder in set_folders:
        set_path = os.path.join(source_dir, set_folder)
        output_set_path = os.path.join(output_dir, set_folder)
        os.makedirs(output_set_path, exist_ok=True)

        # Process each word folder within the set folder
        word_folders = [folder for folder in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, folder))]
        for word_folder in word_folders:
            print(f"Processing word folder: {word_folder} in set {set_folder}")
            process_word_folder(word_folder, set_path, output_set_path)

    print("Processing completed for all word folders in all sets.")


# Run the main function
if __name__ == "__main__":
    main()
