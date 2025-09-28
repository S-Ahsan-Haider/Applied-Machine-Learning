import cv2
import mediapipe as mp

# Initialize MediaPipe's hand tracking model.
# The `mp.solutions.hands` module provides the hand tracking solution.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Use MediaPipe's drawing utilities to visualize the landmarks and connections.
mp_drawing = mp.solutions.drawing_utils

# Finger tip landmark IDs
# The numbers correspond to the specific points on the hand model.
# Index: 8, Middle: 12, Ring: 16, Pinky: 20
# Thumb: 4
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# The webcam index. Change to 1, 2, etc., if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Main loop to capture and process frames from the webcam.
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a more natural, mirror-like view.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB, as MediaPipe requires RGB input.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the hand tracking model.
    # The `process` method returns hand landmarks.
    results = hands.process(rgb_frame)

    finger_count = 0

    # If hands are detected in the frame...
    if results.multi_hand_landmarks:
        # Loop through each detected hand.
        for hand_landmarks in results.multi_hand_landmarks:
            # We will use the y-coordinates of the finger tips and knuckles
            # to determine if a finger is raised.
            
            # Check for the thumb. The thumb is special and its position
            # is relative to the index finger.
            thumb_tip_y = hand_landmarks.landmark[thumb_tip].y
            thumb_mcp_x = hand_landmarks.landmark[2].x # Base of thumb
            thumb_tip_x = hand_landmarks.landmark[thumb_tip].x
            
            # The thumb is counted as up if its tip is to the left of its base (for a right hand, flipped).
            if thumb_tip_x < thumb_mcp_x:
                finger_count += 1
            
            # Check the other four fingers.
            for tip_id in finger_tips:
                tip_y = hand_landmarks.landmark[tip_id].y
                mcp_y = hand_landmarks.landmark[tip_id - 2].y # The knuckle landmark for that finger
                
                # If the finger tip is above the knuckle, it's considered raised.
                if tip_y < mcp_y:
                    finger_count += 1

            # Draw the hand landmarks on the original frame.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

    # Display the final finger count on the frame.
    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the processed frame.
    cv2.imshow('MediaPipe Finger Detector', frame)

    # Press 'q' to exit the application.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
