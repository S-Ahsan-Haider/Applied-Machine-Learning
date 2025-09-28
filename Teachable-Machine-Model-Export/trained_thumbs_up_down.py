import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('model/keras_model.h5')

# Load the labels
with open('model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Read the video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 224x224
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert to numpy array and normalize
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the result on the video feed
    label = f"{class_name.upper()}: {confidence_score:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow('Thumbs Up/Thumbs Down Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()