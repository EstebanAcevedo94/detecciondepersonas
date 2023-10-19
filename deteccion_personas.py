import cv2
from ultralytics import YOLO

# Load the YOLOv8 model for person detection
person_model = YOLO('yolov8n.pt')

# Open the video capture from the camera
cap = cv2.VideoCapture(0)

# Configura la resolución a 1080p
cap.set(3, 1920)  # Ancho
cap.set(4, 1080)  # Alto

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = person_model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Person Detection", annotated_frame)

        # Break the loop if 'esc' is pressed
        k = cv2.waitKey(1)

        if k == 27:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
