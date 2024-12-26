import h5py
import os
import numpy as np
import cv2

def convert_h5_to_png(h5_file_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(h5_file_path, 'r') as f:
        # Assuming 'data' is the key where images are stored
        images = f['data'][:]  # Adjust based on your HDF5 structure

        for i in range(images.shape[0]):
            image = images[i]  # Get each image
            image = (image * 255).astype(np.uint8)  # Convert to uint8 if normalized
            output_path = os.path.join(output_dir, f'image_{i + 1}.png')
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Save as PNG

if __name__ == "__main__":
    h5_image_path = 'TestData/img/image_1.h5'  # Change this to your HDF5 file path
    output_directory = 'output_images'  # Output directory for PNG files
    convert_h5_to_png(h5_image_path, output_directory)
