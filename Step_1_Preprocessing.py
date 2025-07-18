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
            plt.show()
        
        # 2. Apply a blur to help HoughCircles (reduces noise)
        blurred_img = cv2.medianBlur(img, 5)
        
        # return resized_img

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