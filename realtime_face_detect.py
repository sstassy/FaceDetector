import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'facetracker.h5'
facetracker = load_model(model_path)

# Start video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to focus area (adjust as needed)
    frame = frame[50:500, 50:500, :]

    # Preprocess frame for prediction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    input_tensor = np.expand_dims(resized / 255.0, axis=0)

    # Run prediction
    yhat = facetracker.predict(input_tensor)
    confidence = yhat[0][0]
    sample_coords = yhat[1][0]

    if confidence > 0.5:
        start_point = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))

        # Draw bounding box
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

        # Draw label box
        label_start = tuple(np.add(start_point, [0, -30]))
        label_end = tuple(np.add(start_point, [80, 0]))
        cv2.rectangle(frame, label_start, label_end, (255, 0, 0), -1)

        # Draw text label
        label_pos = tuple(np.add(start_point, [0, -5]))
        cv2.putText(frame, 'face', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Real-Time Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
