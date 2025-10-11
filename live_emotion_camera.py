
import cv2
import numpy as np
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the baseline model (85.92% accuracy)
model = load_model('thermal_emotion_baseline_model.h5')
print("Baseline emotion model loaded successfully!")

# Define your emotion classes (update if needed)
emotion_classes = ['angry', 'happy', 'natural', 'sad', 'surprise']

IMG_SIZE = (128, 128)

# Buffer for temporal smoothing
BUFFER_SIZE = 30
emotion_buffer = deque(maxlen=BUFFER_SIZE)

cap = cv2.VideoCapture(0)
prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Calculate FPS every 10 frames
    if frame_count % 10 == 0:
        curr_time = time.time()
        fps = 10 / (curr_time - prev_time)
        prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    detected_emotions = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, IMG_SIZE)
        img_input = np.expand_dims(face_resized / 255.0, axis=0)
        pred_probs = model.predict(img_input)[0]
        pred = np.argmax(pred_probs)
        label = emotion_classes[pred]
        confidence = float(np.max(pred_probs))
        detected_emotions.append(label)
        # Add to buffer for smoothing
        emotion_buffer.append(label)
        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Calculate average emotional state
    if len(faces) > 0 and len(emotion_buffer) > 0:
        avg_emotion = Counter(emotion_buffer).most_common(1)[0][0]
    else:
        avg_emotion = ''
    # Display results
    cv2.putText(frame, f'Avg Emotion: {avg_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Live Emotion Recognition (85.92% Accuracy)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
