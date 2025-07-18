'''
Step 1: Preprocessing - Extracting the Square from the Circular Image

This is the most critical initial step. We'll write a Python script using OpenCV to process a single image first, then extend it to process all images in your dataset.

Goal: Take an image like the one you showed, identify the circular rock region, crop it to a square that tightly encloses the circle, and then resize it to a standard dimension.
'''

#‌Import libraries 
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def preprocess_image(image_path, output_size=(256, 256), plot_steps=False):
    """
    Loads an image, robustly identifies the main circular rock region,
    then crops the LARGEST INSCRIBED SQUARE from WITHIN that circle,
    and resizes it. This version aims to remove all black borders and
    circular outlines, yielding only the internal rock structure.

    Args:
        image_path (str): Path to the input image.
        output_size (tuple): Desired (width, height) for the output square image.
        plot_steps (bool): If True, plots intermediate steps for visualization.

    Returns:
        numpy.ndarray: The preprocessed square image, or None if processing fails.
    """
    try:
        # 1. Load the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # 1.1. 
        original_h, original_w = img.shape[:2]

        # 1.2. plot
        if plot_steps:
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 5, 1)
            plt.title("Original Image")
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            # plt.show()
        
        # 2. Apply a blur to help HoughCircles (reduces noise)
        blurred_img = cv2.medianBlur(img, 5)

        # 3. Detect circles using HoughCircles
        # Parameters for HoughCircles often need fine-tuning per dataset
        # dp=1: Inverse ratio of the accumulator resolution to the image resolution.
        # minDist: Minimum distance between the centers of the detected circles.
        # param1: Upper threshold for the Canny edge detector (internal).
        # param2: Accumulator threshold for the circle centers at the detection stage.
        # minRadius/maxRadius: Expected range of radii of circles.
        circles = cv2.HoughCircles(
            blurred_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=original_h // 8, # Minimum distance between centers
            param1=100, # Canny upper threshold
            param2=30,  # Accumulator threshold
            minRadius=original_h // 4, # Minimum radius
            maxRadius=original_h // 2 # Maximum radius
        )

        # 3.1. If circles is None
        if circles is None:
            print(f"No circles found in {image_path}. Skipping.")
            return None

        # 3.2. Take the first (most prominent) detected circle
        # Ensure coordinates are integers for pixel indexing and calculations
        x_circle, y_circle, r_circle = circles[0, 0].astype(int)

        # 3.3. Ensure radius is not too small (avoids issues with tiny detections)
        if r_circle < 20: # Increased threshold for a meaningful inner square
            print(f"Warning: Detected radius too small ({r_circle}) for {image_path}. Skipping.")
            return None

        # 3.4. --- Visualizing the detected circle (for plot_steps) ---
        if plot_steps:
            img_with_circle = img.copy()
            cv2.circle(img_with_circle, (x_circle, y_circle), r_circle, 255, 2) # Draw circle
            plt.subplot(1, 5, 2)
            plt.title("Detected Circle")
            plt.imshow(img_with_circle, cmap='gray')
            plt.axis('off')
            # plt.show()

        # 4. Calculate the side length of the largest square inscribed in the circle
        # Side length 's' of a square inscribed in a circle with radius 'r' is s = r * sqrt(2)
        # However, for pixel coordinates, we often just use 2 * r / sqrt(2) = sqrt(2) * r
        # More robustly: the corners of the inscribed square are (x +/- r/sqrt(2), y +/- r/sqrt(2))
        # So the side length of the square is 2 * (r / sqrt(2)) = 2 * r / 1.414 = approx 1.414 * r
        # Let's define the side length as 's'
        s = int(r_circle * np.sqrt(2) / 2) * 2 # Ensure it's an even number if preferred, or just int(r_circle * np.sqrt(2)) for total side

        # 4.1. A simpler way to get half_side for calculating top-left from center:
        half_side = int(r_circle / np.sqrt(2))
        side_length = 2 * half_side

        # 4.2. If side length
        if side_length <= 0:
            print(f"Error: Calculated side_length for inner square is zero or negative ({side_length}). Skipping.")
            return None

        # 5. Calculate the coordinates of the inscribed square
        # Top-left corner (x1, y1) and bottom-right corner (x2, y2)
        x1 = x_circle - half_side
        y1 = y_circle - half_side
        x2 = x_circle + half_side
        y2 = y_circle + half_side

        # 5.1. Ensure these coordinates are within the original image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_w, x2)
        y2 = min(original_h, y2)

        # 5.2. Adjust dimensions to be exact side_length if possible,
        # otherwise, this crop might be slightly off-center if it hit image bounds.
        # For simplicity, if it hits bounds, we just crop what's available.
        # This will result in a slightly smaller-than-ideal inner square, but still valid.
        actual_cropped_width = x2 - x1
        actual_cropped_height = y2 - y1

        # 5.3. If actual cropped size
        if actual_cropped_width <= 0 or actual_cropped_height <= 0:
            print(f"Error: Invalid actual crop dimensions ({actual_cropped_width}x{actual_cropped_height}) for {image_path}. Skipping.")
            return None

        # 6. Crop the original image to this inscribed square
        cropped_img = img[y1:y2, x1:x2]

        # 6.1. If crop size
        if cropped_img.size == 0:
            print(f"Error: Cropped image is empty after slicing for {image_path}. Skipping.")
            return None

        # 6.2. Plot
        if plot_steps:
            plt.subplot(1, 5, 3)
            plt.title(f"Inscribed Cropped Image ({cropped_img.shape[1]}x{cropped_img.shape[0]})")
            plt.imshow(cropped_img, cmap='gray')
            plt.axis('off')

            # Optional: Visualize the square on the original image
            img_with_square = img.copy()
            cv2.rectangle(img_with_square, (x1, y1), (x2, y2), 255, 2)
            plt.subplot(1, 5, 4)
            plt.title("Original with Inscribed Square")
            plt.imshow(img_with_square, cmap='gray')
            plt.axis('off')
        
        # 7. Resize the cropped image to the desired output size
        resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)

        # 7.1. Plot
        if plot_steps:
            plt.subplot(1, 5, 5)
            plt.title(f"Resized Output ({output_size[0]}x{output_size[1]})")
            plt.imshow(resized_img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return resized_img

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        print(3)
        return None


# --- Example usage (replace with your actual sample image path) ---
sample_image_path = 'data/sample_data/sample_image.png' # Update this to your actual path
# Make sure Figure_3.png is named as sample_image.png or change path above
processed_sample_img = preprocess_image(sample_image_path, plot_steps=True)

if processed_sample_img is not None:
    print(f"Processed image shape: {processed_sample_img.shape}")
else:
    print("Image processing failed.")