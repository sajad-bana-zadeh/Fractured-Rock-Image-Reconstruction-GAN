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
    Loads an image, crops the circular region to a square, and resizes it.

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

        # 2. plot image if plot_steps == true
        if plot_steps:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.show()

        return None

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