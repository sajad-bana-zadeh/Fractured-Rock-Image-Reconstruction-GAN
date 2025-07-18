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
    crops it to a square, and resizes it. This version aims to exclude annotations.

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

        # 1.2. plot image if plot_steps == true
        if plot_steps:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            # plt.show()
        
        # 2. Apply a blur to help HoughCircles (reduces noise)
        blurred_img = cv2.medianBlur(img, 5) # Median blur is good for preserving edges while removing noise
        
        # 3. Detect circles using HoughCircles
        # dp: inverse ratio of the accumulator resolution to the image resolution.
        # minDist: minimum distance between the centers of the detected circles.
        # param1: upper threshold for the Canny edge detector (internal).
        # param2: accumulator threshold for the circle centers at the detection stage.
        # minRadius/maxRadius: expected range of radii of circles.
        circles = cv2.HoughCircles(
            blurred_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=img.shape[0] // 8, # Minimum distance between centers (e.g., 1/8th of image height)
            param1=100, # Canny upper threshold
            param2=30,  # Accumulator threshold
            minRadius=img.shape[0] // 4, # Minimum radius (e.g., 1/4th of image height)
            maxRadius=img.shape[0] // 2 # Maximum radius (e.g., 1/2 of image height, as it's a diameter)
        )

        # 3.1. if circle is find
        if circles is None:
            print(f"No circles found in {image_path}. Skipping.")
            return None

        # 3.2. Ensure we only consider the most prominent circle
        circles = np.uint16(np.around(circles))
        # Take the first and likely most confident circle detected
        x, y, r = circles[0, 0]

        # 4. Get the minimum enclosing circle for the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Ensure radius is not zero or too small
        if radius < 5: # A small threshold to avoid issues with tiny artifacts
            print(f"Warning: Radius too small ({radius:.2f}) for {image_path}. Skipping.")
            return None
        
        # 5. Calculate the bounding box for the square crop
        center_x, center_y = int(x), int(y)
        radius = int(radius) # Cast to int for pixel coordinates

        # 5.1. Determine the square's top-left corner and side length
        # We want to crop a square of side_length = 2 * radius around the center
        crop_size = 2 * radius
        x_start = max(0, center_x - radius)
        y_start = max(0, center_y - radius)

        # 5.2. Ensure crop_size does not exceed image boundaries
        # This part ensures we don't go out of bounds when calculating x_end, y_end
        x_end = min(img.shape[1], x_start + crop_size)
        y_end = min(img.shape[0], y_start + crop_size)

        # 5.3. Adjust x_start/y_start if the calculated end goes beyond image boundary
        # This helps in cases where the circle might be very close to an edge
        if (x_end - x_start) < crop_size:
            x_start = max(0, x_end - crop_size)
        if (y_end - y_start) < crop_size:
            y_start = max(0, y_end - crop_size)
        
        # 6. Crop the image
        cropped_img = img[y_start:y_end, x_start:x_end]

        #‌ 6.2. plot image if plot_steps == true
        if plot_steps:
            plt.subplot(1, 4, 3)
            plt.title(f"Cropped Image ({cropped_img.shape[1]}x{cropped_img.shape[0]})")
            plt.imshow(cropped_img, cmap='gray')
            plt.axis('off')
            # plt.show()
    
        # 7. Resize the cropped image to the desired output size
        resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)

        #‌ 7.2. plot image if plot_steps == true
        if plot_steps:
            plt.subplot(1, 4, 4)
            plt.title(f"Resized Image ({output_size[0]}x{output_size[1]})")
            plt.imshow(resized_img, cmap='gray')
            plt.axis('off')
            plt.show()

        return resized_img

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        print(3)
        return None


# --- How to test this function with your sample image ---
# 1. set your sample image (sample_image.png)
#    into the same directory as your Python script, "Fractured-Rock-Image-Reconstruction-GAN/data/sample_data/sample_image.png", or provide its full path.
# 2. Replace 'sample_image.png' with the actual file name.

# Example usage:
if __name__ == '__main__':
    sample_image_path = 'data/sample_data/sample_image.png'
    processed_sample_img = preprocess_image(sample_image_path, plot_steps=True)

    if processed_sample_img is not None:
        print(f"Processed image shape: {processed_sample_img.shape}")
    else:
        print("Image processing failed.")