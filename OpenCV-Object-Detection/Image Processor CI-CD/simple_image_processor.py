import cv2
import os

# Define the input and output file names
input_image_path = 'input.jpg'
output_image_path = 'output_grayscale.jpg'

# Check if the input image exists
if not os.path.exists(input_image_path):
    print(f"Error: The file '{input_image_path}' was not found.")
    print("Please place a sample image named 'input.jpg' in the same directory.")
else:
    try:
        # Read the image from the specified path
        image = cv2.imread(input_image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image to the specified output path
        cv2.imwrite(output_image_path, grayscale_image)

        print(f"Successfully converted '{input_image_path}' to grayscale and saved as '{output_image_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
