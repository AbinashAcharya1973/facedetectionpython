import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Database setup
def initialize_database():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            face BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_face_to_db(face_image):
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()

    # Convert face to binary format
    _, buffer = cv2.imencode('.jpg', face_image)
    face_blob = buffer.tobytes()

    # Insert data into the database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO Faces (datetime, face) VALUES (?, ?)", (timestamp, face_blob))
    conn.commit()
    conn.close()

# Initialize database
initialize_database()

# Load the pre-trained model and prototxt file
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start video capture (use 0 for webcam)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:Subodh@123@192.168.1.64:554")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 300x300 and normalize pixel values
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and get detections
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the detected face
            face = frame[startY:endY, startX:endX]
            if face.size > 0:  # Ensure the face region is valid
                save_face_to_db(face)

            # Draw a rectangle and label on the frame
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Face Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
