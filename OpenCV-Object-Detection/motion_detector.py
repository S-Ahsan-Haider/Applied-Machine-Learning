# motion_detector.py

import cv2
import time

# --- Setup ---
# Initialize the webcam to capture video.
# The argument '0' refers to the default camera. You can change this if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Allow the camera to warm up for a moment to get a stable first frame.
# This helps prevent false motion detection at the start.
time.sleep(2)

# Read the first frame and convert it to grayscale for comparison.
# This will be our "background" reference frame.
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read from webcam. Exiting.")
    exit()

# Convert the reference frame to grayscale and apply a Gaussian blur.
# Grayscale simplifies the image, and blurring reduces noise, making motion detection more reliable.
gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray_frame1 = cv2.GaussianBlur(gray_frame1, (21, 21), 0)

print("Motion detector is ready. Press 'q' to quit.")

# --- Main Loop ---
# This loop continuously reads frames and checks for motion.
while True:
    # Read the next frame from the camera.
    ret, frame2 = cap.read()
    if not ret:
        print("Error: Could not read next frame. Exiting.")
        break

    # Convert the new frame to grayscale and blur it, just like the reference frame.
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.GaussianBlur(gray_frame2, (21, 21), 0)

    # Calculate the absolute difference between the two frames.
    # Pixels that have changed significantly will have a high difference value.
    frame_delta = cv2.absdiff(gray_frame1, gray_frame2)

    # Apply a threshold to the difference image.
    # This turns the difference into a binary image, where white pixels are changes and black pixels are not.
    # The threshold value (30) can be adjusted to make the detector more or less sensitive.
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in small gaps and make motion more prominent.
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image.
    # These contours represent the boundaries of the moving objects.
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the detected contours.
    for contour in contours:
        # If a contour is too small, it's probably just noise. Skip it.
        # The contour area (1000) can be adjusted.
        if cv2.contourArea(contour) < 1000:
            continue

        # Get the bounding box coordinates for the contour.
        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw a green rectangle around the moving object on the original color frame.
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the current frame with the motion detection rectangles.
    cv2.imshow('Motion Detector', frame2)

    # Set the current frame as the new reference frame for the next iteration.
    gray_frame1 = gray_frame2

    # Check for the 'q' key press to quit the application.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
