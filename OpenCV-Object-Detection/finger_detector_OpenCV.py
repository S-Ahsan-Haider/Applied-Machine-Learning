import cv2
import numpy as np

# A lightweight script to detect and count fingers using a webcam.
# The method uses background subtraction and contour analysis to achieve the results.

# --- Helper Functions ---
def find_largest_contour(contours):
    """Finds the largest contour from a list of contours."""
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour

# --- Main Application ---
def main():
    # Use CAP_DSHOW for more stable webcam access on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set a standard resolution to ensure consistency between different images.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use by another application.")
        return

    roi_top, roi_left = 100, 350
    roi_bottom, roi_right = 350, 600

    # Initialize a background subtractor to remove the unnecessary things from images.
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    
    is_bg_ready = False
    background_frames = 0
    
    print("Hold hand still for background caliberation")

    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                print("Warning: Failed to read a valid frame. Skipping.")
                continue

            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
            
            roi = frame[roi_top:roi_bottom, roi_left:roi_right]

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            if not is_bg_ready:
                if background_frames < 30:
                    bg_subtractor.apply(gray_roi, learningRate=0.01)
                    background_frames += 1
                else:
                    is_bg_ready = True
                    print("Background calibrated. You can now place your hand in the green box.")
                
                cv2.putText(frame, "Calibrating...", (roi_left + 10, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                fg_mask = bg_subtractor.apply(gray_roi, learningRate=0)
                _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    hand_contour = find_largest_contour(contours)
                    
                    if hand_contour is not None and cv2.contourArea(hand_contour) > 500:
                        cv2.drawContours(roi, [hand_contour + (roi_left, roi_top)], -1, (255, 0, 0), 2)
                        
                        finger_count = 0
                        
                        # Use a try-except block to gracefully handle the convexity defects error.
                        try:
                            hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
                            defects = cv2.convexityDefects(hand_contour, hull_indices)

                            if defects is not None:
                                for i in range(defects.shape[0]):
                                    s, e, f, d = defects[i, 0]
                                    start = tuple(hand_contour[s][0])
                                    end = tuple(hand_contour[e][0])
                                    far = tuple(hand_contour[f][0])
                                    
                                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                                    
                                    if 2 * b * c != 0:
                                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                                    else:
                                        angle = np.pi
                                        
                                    if angle <= np.pi / 2:
                                        finger_count += 1
                                        cv2.circle(roi, far, 5, [0, 0, 255], -1)
                            
                            # Heuristic: the number of fingers is often one more than the number of defects.
                            # For a straight hand, there are 4 defects (spaces between fingers) and 5 fingers.
                            # For a single raised finger, there are no defects, but the logic should show one.
                            if finger_count > 0:
                                finger_count += 1
                            
                        except cv2.error as e:
                            # If a cv2.error occurs, just print a warning and continue.
                            print(f"Warning: OpenCV error during convexity defects. Skipping finger counting for this frame. Details: {e}")
                            finger_count = "N/A"
                        
                        cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No hand detected", (roi_left + 10, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Finger Detector', frame)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
