import cv2
import time

# Replace with your webcam index (0 is usually the default webcam)
webcam_index = 0

# Open the webcam
#cap = cv2.VideoCapture(webcam_index)
cap = cv2.VideoCapture("rtsp://admin:Subodh@123@192.168.1.64:554")

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Frame failure counter
max_retries = 5
retry_count = 0

while True:
    ret, frame = cap.read()

    # Handle frame reading failure
    if not ret:
        print("Failed to grab frame. Retrying...")
        retry_count += 1
        time.sleep(1)  # Wait before retrying
        if retry_count >= max_retries:
            print("Error: Maximum retry limit reached. Exiting...")
            break
        continue

    retry_count = 0  # Reset retry counter if a frame is successfully grabbed

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()