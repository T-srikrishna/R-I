"""
Multi-Face Detection Test
========================

Simple test script to verify multiple face detection is working.
"""

import cv2
import numpy as np

def test_multi_face_detection():
    """Test if face detection can find multiple faces"""
    
    print("🔍 Testing Multi-Face Detection")
    print("=" * 40)
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("📹 Camera opened successfully")
    print("🎯 Position multiple faces in front of camera")
    print("⚙️  Testing different detection parameters...")
    print("🎮 Press 'q' to quit, 's' to switch parameters")
    
    # Different parameter sets for testing
    param_sets = [
        {
            'name': 'Conservative',
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (60, 60)
        },
        {
            'name': 'Balanced',
            'scaleFactor': 1.05,
            'minNeighbors': 3,
            'minSize': (40, 40)
        },
        {
            'name': 'Aggressive',
            'scaleFactor': 1.03,
            'minNeighbors': 2,
            'minSize': (30, 30)
        }
    ]
    
    current_params = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Current parameter set
            params = param_sets[current_params]
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'],
                minSize=params['minSize'],
                maxSize=(400, 400)
            )
            
            # Draw results
            display_frame = frame.copy()
            
            # Draw each detected face
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, (x, y, w, h) in enumerate(faces):
                color = colors[i % len(colors)]
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display_frame, f"Face {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw info
            cv2.putText(display_frame, f"Faces Detected: {len(faces)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"Parameters: {params['name']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"Scale: {params['scaleFactor']}, Neighbors: {params['minNeighbors']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "Press 's' to switch parameters, 'q' to quit", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Multi-Face Detection Test', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_params = (current_params + 1) % len(param_sets)
                print(f"🔄 Switched to {param_sets[current_params]['name']} parameters")
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Multi-face detection test completed")

if __name__ == "__main__":
    test_multi_face_detection()