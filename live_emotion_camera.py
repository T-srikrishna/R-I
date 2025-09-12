import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('thermal_emotion_model.h5')

# Define your emotion classes (update if needed)
emotion_classes = ['angry', 'happy', 'natural', 'sad', 'surpise']

IMG_SIZE = (128, 128)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, IMG_SIZE)
    img_input = np.expand_dims(img / 255.0, axis=0)
    pred = np.argmax(model.predict(img_input), axis=1)[0]
    label = emotion_classes[pred]
    cv2.putText(frame, f'Predicted Emotion: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Webcam Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
