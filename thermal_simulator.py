"""
Simple Thermal Camera Simulator
===============================

This script simulates a thermal camera using your laptop's regular camera.
It applies thermal-like effects and includes emotion detection + temperature simulation.
Perfect for testing thermal emotion recognition without actual thermal hardware.
"""

import cv2
import numpy as np
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model
from datetime import datetime
import json

class ThermalCameraSimulator:
    """Simulates thermal camera behavior using regular webcam"""
    
    def __init__(self, model_path='thermal_emotion_model_enhanced.h5'):
        # Load the trained model
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure thermal_emotion_model_enhanced.h5 exists in the current directory")
            exit(1)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # System parameters
        self.emotion_classes = ['angry', 'happy', 'natural', 'sad', 'surprise']
        self.img_size = (128, 128)
        
        # Thermal simulation parameters
        self.thermal_colormap = cv2.COLORMAP_JET  # Thermal-like colors
        self.base_temp = 36.5  # Base body temperature
        self.temp_noise = 0.5  # Temperature variation
        
        # Tracking and smoothing
        self.emotion_buffer = deque(maxlen=30)
        self.temperature_buffer = deque(maxlen=30)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        # Data logging
        self.detection_log = []
        
        print("🎥 Thermal Camera Simulator initialized successfully!")
        print("📊 Features: Emotion Detection + Temperature Simulation")
    
    def apply_thermal_effect(self, frame):
        """Apply thermal camera visual effects to regular camera frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast to simulate thermal imaging
        enhanced = cv2.equalizeHist(gray)
        
        # Add some thermal-like noise
        noise = np.random.normal(0, 5, enhanced.shape).astype(np.uint8)
        thermal_gray = cv2.add(enhanced, noise)
        
        # Apply thermal colormap (jet colormap gives blue-red thermal look)
        thermal_colored = cv2.applyColorMap(thermal_gray, self.thermal_colormap)
        
        return thermal_colored, thermal_gray
    
    def simulate_temperature(self, face_region):
        """Simulate body temperature from face region brightness"""
        # Calculate average brightness of face region
        avg_brightness = np.mean(face_region)
        
        # Convert brightness to temperature (simulate thermal camera behavior)
        # Brighter areas = warmer in thermal imaging
        temp_factor = avg_brightness / 255.0
        
        # Simulate body temperature range (35-39°C)
        simulated_temp = self.base_temp + (temp_factor * 2.5) + np.random.normal(0, self.temp_noise)
        
        # Keep within reasonable body temperature range
        simulated_temp = np.clip(simulated_temp, 34.0, 40.0)
        
        return round(simulated_temp, 1)
    
    def detect_emotion_and_temperature(self, frame):
        """Detect emotions and simulate temperature from frame"""
        # Apply thermal effects
        thermal_frame, thermal_gray = self.apply_thermal_effect(frame)
        
        # Try face detection on both original and thermal frames for better results
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces on original frame (often better for multiple faces)
        faces_original = self.face_cascade.detectMultiScale(
            original_gray, 
            scaleFactor=1.05,      # Smaller scale factor for better detection
            minNeighbors=3,        # Lower neighbors threshold for more sensitivity
            minSize=(40, 40),      # Smaller minimum size to catch distant faces
            maxSize=(400, 400),    # Maximum size to avoid false positives
            flags=cv2.CASCADE_SCALE_IMAGE  # Better scaling
        )
        
        # Detect faces on thermal frame as backup
        faces_thermal = self.face_cascade.detectMultiScale(
            thermal_gray, 
            scaleFactor=1.08,      # Slightly different parameters
            minNeighbors=4,        
            minSize=(45, 45),      
            maxSize=(350, 350)
        )
        
        # Combine and filter face detections (remove duplicates)
        all_faces = []
        
        # Add faces from original detection
        for face in faces_original:
            all_faces.append(face)
        
        # Add faces from thermal detection if they don't overlap too much
        for thermal_face in faces_thermal:
            tx, ty, tw, th = thermal_face
            is_duplicate = False
            
            for orig_face in faces_original:
                ox, oy, ow, oh = orig_face
                # Check for significant overlap
                overlap_x = max(0, min(tx + tw, ox + ow) - max(tx, ox))
                overlap_y = max(0, min(ty + th, oy + oh) - max(ty, oy))
                overlap_area = overlap_x * overlap_y
                thermal_area = tw * th
                
                if overlap_area > 0.3 * thermal_area:  # 30% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_faces.append(thermal_face)
        
        # Use the combined face list
        faces = all_faces
        
        detections = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            face_thermal = thermal_gray[y:y+h, x:x+w]
            
            # Prepare face for emotion recognition
            face_resized = cv2.resize(face_img, self.img_size)
            if len(face_resized.shape) == 3:
                face_input = face_resized / 255.0
            else:
                face_input = np.stack([face_resized/255.0] * 3, axis=-1)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Predict emotion
            try:
                pred_probs = self.model.predict(face_input, verbose=0)[0]
                pred_idx = np.argmax(pred_probs)
                emotion = self.emotion_classes[pred_idx]
                confidence = float(np.max(pred_probs))
            except Exception as e:
                print(f"Emotion prediction error: {e}")
                emotion = "unknown"
                confidence = 0.0
            
            # Simulate temperature
            temperature = self.simulate_temperature(face_thermal)
            
            # Store detection
            detection = {
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'temperature': temperature,
                'timestamp': datetime.now()
            }
            
            detections.append(detection)
            
            # Update buffers for smoothing
            self.emotion_buffer.append(emotion)
            self.temperature_buffer.append(temperature)
        
        return thermal_frame, detections
    
    def draw_information(self, thermal_frame, detections):
        """Draw detection information on the frame"""
        display_frame = thermal_frame.copy()
        
        # Draw detection information for each face
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            temperature = detection['temperature']
            
            # Color based on emotion
            emotion_colors = {
                'happy': (0, 255, 0),      # Green
                'natural': (255, 255, 0),  # Yellow
                'angry': (0, 0, 255),      # Red
                'sad': (255, 0, 255),      # Magenta
                'surprise': (0, 255, 255)  # Cyan
            }
            color = emotion_colors.get(emotion, (128, 128, 128))
            
            # Draw bounding box with thicker lines for visibility
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
            
            # Face ID number
            face_id = f"Face {i+1}"
            cv2.putText(display_frame, face_id, (x, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw emotion label
            emotion_text = f"{emotion} ({confidence:.2f})"
            cv2.putText(display_frame, emotion_text, (x, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw temperature
            temp_text = f"Temp: {temperature}°C"
            cv2.putText(display_frame, temp_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Temperature warning
            if temperature > 37.5:
                cv2.putText(display_frame, "HIGH TEMP!", (x, y+h+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif temperature < 35.5:
                cv2.putText(display_frame, "LOW TEMP", (x, y+h+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw system information
        info_y = 30
        
        # FPS
        cv2.putText(display_frame, f"FPS: {self.fps:.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mode indicator
        cv2.putText(display_frame, "THERMAL SIMULATOR", 
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Number of faces detected
        cv2.putText(display_frame, f"Faces Detected: {len(detections)}", 
                   (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Average emotion
        if len(self.emotion_buffer) > 0:
            avg_emotion = Counter(self.emotion_buffer).most_common(1)[0][0]
            cv2.putText(display_frame, f"Avg Emotion: {avg_emotion}", 
                       (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Average temperature
        if len(self.temperature_buffer) > 0:
            avg_temp = np.mean(list(self.temperature_buffer))
            cv2.putText(display_frame, f"Avg Temp: {avg_temp:.1f}°C", 
                       (10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detection count
        cv2.putText(display_frame, f"Total Detections: {len(self.detection_log)}", 
                   (10, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display_frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            curr_time = time.time()
            self.fps = 10 / (curr_time - self.prev_time)
            self.prev_time = curr_time
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thermal_sim_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def export_data(self):
        """Export detection data to JSON file"""
        if not self.detection_log:
            print("❌ No data to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thermal_sim_data_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        export_data = []
        for detection in self.detection_log:
            data = detection.copy()
            data['timestamp'] = data['timestamp'].isoformat()
            export_data.append(data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Data exported: {filename}")
        print(f"📈 Total detections: {len(export_data)}")
    
    def run(self, camera_source=0):
        """Run the thermal camera simulator"""
        print("\n🚀 Starting Thermal Camera Simulator...")
        print("📹 Using camera source:", camera_source)
        print("\n🎮 Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'e' - Export data")
        print("  'r' - Reset data")
        print("  SPACE - Pause/Resume")
        print("\n" + "="*50)
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Could not read frame")
                        break
                    
                    # Calculate FPS
                    self.calculate_fps()
                    
                    # Process frame
                    thermal_frame, detections = self.detect_emotion_and_temperature(frame)
                    
                    # Log detections
                    self.detection_log.extend(detections)
                    
                    # Draw information
                    display_frame = self.draw_information(thermal_frame, detections)
                else:
                    # Show last frame when paused
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Thermal Camera Simulator - Emotion & Temperature', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(display_frame)
                elif key == ord('e'):
                    self.export_data()
                elif key == ord('r'):
                    self.detection_log.clear()
                    self.emotion_buffer.clear()
                    self.temperature_buffer.clear()
                    print("🔄 Data reset")
                elif key == ord(' '):  # Space bar
                    paused = not paused
                    print("⏸️  Paused" if paused else "▶️  Resumed")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n📊 Session Summary:")
            print(f"   Total detections: {len(self.detection_log)}")
            if self.emotion_buffer:
                emotions = Counter(self.emotion_buffer)
                print("   Emotion distribution:", dict(emotions))
            if self.temperature_buffer:
                temps = list(self.temperature_buffer)
                print(f"   Temperature range: {min(temps):.1f}°C - {max(temps):.1f}°C")
            print("\n✅ Thermal Camera Simulator closed")

def main():
    """Main function"""
    print("🌡️  Thermal Camera Simulator")
    print("=" * 50)
    print("🎯 Simulates thermal camera using regular laptop camera")
    print("🧠 Includes emotion detection + temperature simulation")
    print("🎨 Applies thermal-like visual effects")
    print("=" * 50)
    
    # Initialize and run simulator
    simulator = ThermalCameraSimulator()
    simulator.run()

if __name__ == "__main__":
    main()