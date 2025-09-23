"""
Multi-Person Thermal Monitoring System
======================================

Advanced thermal simulator designed for multi-person scenarios like:
- Airport/building entry screening
- Multiple people walking through thermal checkpoints
- Group monitoring in offices/public spaces
- Real-time health screening of crowds

This simulator prepares you for real thermal camera deployments.
"""

import cv2
import numpy as np
import time
from collections import deque, Counter, defaultdict
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import uuid

class PersonTracker:
    """Track individual persons across frames"""
    def __init__(self, person_id, bbox, emotion, confidence, temperature):
        self.id = person_id
        self.bbox_history = deque([bbox], maxlen=10)
        self.emotion_history = deque([emotion], maxlen=5)
        self.confidence_history = deque([confidence], maxlen=5)
        self.temperature_history = deque([temperature], maxlen=5)
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.frame_count = 1
        
    def update(self, bbox, emotion, confidence, temperature):
        """Update person with new detection"""
        self.bbox_history.append(bbox)
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        self.temperature_history.append(temperature)
        self.last_seen = datetime.now()
        self.frame_count += 1
    
    def get_current_bbox(self):
        return self.bbox_history[-1]
    
    def get_average_emotion(self):
        emotions = list(self.emotion_history)
        if not emotions:
            return "unknown"
        return Counter(emotions).most_common(1)[0][0]
    
    def get_average_temperature(self):
        temps = list(self.temperature_history)
        if not temps:
            return 0.0
        return np.mean(temps)
    
    def get_average_confidence(self):
        confs = list(self.confidence_history)
        if not confs:
            return 0.0
        return np.mean(confs)

class MultiPersonThermalSystem:
    """Multi-person thermal monitoring system"""
    
    def __init__(self, model_path='thermal_emotion_model_enhanced.h5'):
        # Load the trained model
        try:
            self.model = load_model(model_path)
            print(f"✅ Enhanced Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # System parameters
        self.emotion_classes = ['angry', 'happy', 'natural', 'sad', 'surprise']
        self.img_size = (128, 128)
        
        # Multi-person tracking
        self.active_persons = {}  # person_id: PersonTracker
        self.next_person_id = 1
        self.max_tracking_distance = 100  # pixels
        self.person_timeout = 3.0  # seconds before removing lost person
        
        # Thermal simulation
        self.thermal_colormap = cv2.COLORMAP_JET
        self.base_temp = 36.5
        self.temp_noise = 0.3
        
        # Statistics
        self.total_persons_seen = 0
        self.current_person_count = 0
        self.session_stats = {
            'high_temp_alerts': 0,
            'low_temp_alerts': 0,
            'emotion_alerts': 0,
            'total_detections': 0
        }
        
        # Alert thresholds
        self.high_temp_threshold = 37.8
        self.low_temp_threshold = 35.5
        self.alert_emotions = ['angry', 'sad']
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        print("🎥 Multi-Person Thermal System initialized!")
        print("👥 Ready for crowd monitoring and screening")
    
    def apply_advanced_thermal_effect(self, frame):
        """Enhanced thermal effects for better simulation"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for thermal simulation
        # 1. Histogram equalization for better contrast
        enhanced = cv2.equalizeHist(gray)
        
        # 2. Gaussian blur to simulate thermal diffusion
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 3. Edge enhancement to highlight body heat boundaries
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        enhanced_edges = np.uint8(np.absolute(laplacian))
        thermal_enhanced = cv2.addWeighted(blurred, 0.8, enhanced_edges, 0.2, 0)
        
        # 4. Add realistic thermal noise
        noise = np.random.normal(0, 8, thermal_enhanced.shape).astype(np.uint8)
        thermal_gray = cv2.add(thermal_enhanced, noise)
        
        # 5. Apply thermal colormap
        thermal_colored = cv2.applyColorMap(thermal_gray, self.thermal_colormap)
        
        return thermal_colored, thermal_gray
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        
        # Calculate total area
        area1 = w1 * h1
        area2 = w2 * h2
        total_area = area1 + area2 - overlap_area
        
        return overlap_area / total_area if total_area > 0 else 0
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate center distance between bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def simulate_body_temperature(self, face_region, person_id=None):
        """Simulate realistic body temperature"""
        avg_brightness = np.mean(face_region)
        
        # Base temperature calculation
        temp_factor = avg_brightness / 255.0
        base_temp = self.base_temp + (temp_factor * 3.0)
        
        # Add person-specific variation (simulate individual differences)
        if person_id:
            np.random.seed(person_id)  # Consistent per person
            person_variation = np.random.normal(0, 0.5)
            base_temp += person_variation
        
        # Add temporal noise
        temporal_noise = np.random.normal(0, self.temp_noise)
        simulated_temp = base_temp + temporal_noise
        
        # Clamp to realistic range
        simulated_temp = np.clip(simulated_temp, 34.0, 42.0)
        
        return round(simulated_temp, 1)
    
    def detect_and_track_persons(self, frame):
        """Detect faces and track persons across frames"""
        # Apply thermal effects
        thermal_frame, thermal_gray = self.apply_advanced_thermal_effect(frame)
        
        # Multi-scale face detection for better results
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Primary detection (conservative)
        faces_primary = self.face_cascade.detectMultiScale(
            original_gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(50, 50),
            maxSize=(300, 300)
        )
        
        # Secondary detection (aggressive for distant faces)
        faces_secondary = self.face_cascade.detectMultiScale(
            thermal_gray,
            scaleFactor=1.03,
            minNeighbors=2,
            minSize=(35, 35),
            maxSize=(400, 400)
        )
        
        # Combine detections and remove duplicates
        all_faces = list(faces_primary)
        for face in faces_secondary:
            is_duplicate = False
            for existing_face in all_faces:
                if self.calculate_overlap(face, existing_face) > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_faces.append(face)
        
        current_detections = []
        
        # Process each detected face
        for bbox in all_faces:
            x, y, w, h = bbox
            
            # Extract face regions
            face_img = frame[y:y+h, x:x+w]
            face_thermal = thermal_gray[y:y+h, x:x+w]
            
            # Emotion recognition
            try:
                face_resized = cv2.resize(face_img, self.img_size)
                if len(face_resized.shape) == 3:
                    face_input = face_resized / 255.0
                else:
                    face_input = np.stack([face_resized/255.0] * 3, axis=-1)
                face_input = np.expand_dims(face_input, axis=0)
                
                pred_probs = self.model.predict(face_input, verbose=0)[0]
                pred_idx = np.argmax(pred_probs)
                emotion = self.emotion_classes[pred_idx]
                confidence = float(np.max(pred_probs))
            except Exception as e:
                emotion = "unknown"
                confidence = 0.0
            
            current_detections.append({
                'bbox': bbox,
                'emotion': emotion,
                'confidence': confidence,
                'face_thermal': face_thermal
            })
        
        # Update person tracking
        self.update_person_tracking(current_detections)
        
        return thermal_frame, list(self.active_persons.values())
    
    def update_person_tracking(self, detections):
        """Update person tracking with new detections"""
        current_time = datetime.now()
        
        # Match detections to existing persons
        unmatched_detections = list(detections)
        updated_persons = set()
        
        for person_id, person in list(self.active_persons.items()):
            best_match = None
            best_distance = float('inf')
            
            # Find closest detection to this person's last position
            last_bbox = person.get_current_bbox()
            
            for i, detection in enumerate(unmatched_detections):
                distance = self.calculate_distance(last_bbox, detection['bbox'])
                if distance < best_distance and distance < self.max_tracking_distance:
                    best_distance = distance
                    best_match = i
            
            # Update person if match found
            if best_match is not None:
                detection = unmatched_detections.pop(best_match)
                
                # Simulate temperature for this specific person
                temperature = self.simulate_body_temperature(
                    detection['face_thermal'], 
                    person_id
                )
                
                person.update(
                    detection['bbox'],
                    detection['emotion'],
                    detection['confidence'],
                    temperature
                )
                updated_persons.add(person_id)
        
        # Remove persons not seen for too long
        for person_id in list(self.active_persons.keys()):
            if person_id not in updated_persons:
                person = self.active_persons[person_id]
                time_since_seen = (current_time - person.last_seen).total_seconds()
                if time_since_seen > self.person_timeout:
                    del self.active_persons[person_id]
        
        # Add new persons for unmatched detections
        for detection in unmatched_detections:
            person_id = self.next_person_id
            self.next_person_id += 1
            
            temperature = self.simulate_body_temperature(
                detection['face_thermal'], 
                person_id
            )
            
            person = PersonTracker(
                person_id,
                detection['bbox'],
                detection['emotion'],
                detection['confidence'],
                temperature
            )
            
            self.active_persons[person_id] = person
            self.total_persons_seen += 1
        
        # Update statistics
        self.current_person_count = len(self.active_persons)
        self.session_stats['total_detections'] += len(detections)
    
    def check_alerts(self, persons):
        """Check for alert conditions across all persons"""
        alerts = []
        
        for person in persons:
            temp = person.get_average_temperature()
            emotion = person.get_average_emotion()
            
            # Temperature alerts
            if temp > self.high_temp_threshold:
                alerts.append({
                    'type': 'HIGH_TEMP',
                    'person_id': person.id,
                    'value': temp,
                    'message': f"Person {person.id}: HIGH TEMP {temp:.1f}°C"
                })
                self.session_stats['high_temp_alerts'] += 1
            
            elif temp < self.low_temp_threshold:
                alerts.append({
                    'type': 'LOW_TEMP',
                    'person_id': person.id,
                    'value': temp,
                    'message': f"Person {person.id}: LOW TEMP {temp:.1f}°C"
                })
                self.session_stats['low_temp_alerts'] += 1
            
            # Emotion alerts
            if emotion in self.alert_emotions:
                alerts.append({
                    'type': 'EMOTION',
                    'person_id': person.id,
                    'value': emotion,
                    'message': f"Person {person.id}: {emotion.upper()} emotion"
                })
                self.session_stats['emotion_alerts'] += 1
        
        return alerts
    
    def draw_multi_person_display(self, thermal_frame, persons, alerts):
        """Draw comprehensive multi-person information"""
        display_frame = thermal_frame.copy()
        
        # Color palette for different persons
        person_colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
        
        # Draw each person
        for i, person in enumerate(persons):
            x, y, w, h = person.get_current_bbox()
            color = person_colors[i % len(person_colors)]
            
            # Person bounding box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
            
            # Person ID
            cv2.putText(display_frame, f"Person {person.id}", 
                       (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Emotion (averaged)
            emotion = person.get_average_emotion()
            confidence = person.get_average_confidence()
            cv2.putText(display_frame, f"{emotion} ({confidence:.2f})", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Temperature (averaged)
            temp = person.get_average_temperature()
            cv2.putText(display_frame, f"Temp: {temp:.1f}°C", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Tracking duration
            duration = (datetime.now() - person.first_seen).total_seconds()
            cv2.putText(display_frame, f"Track: {duration:.1f}s", 
                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # System information panel
        info_x, info_y = 10, 30
        panel_width = 400
        
        # Semi-transparent background for info panel
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # System stats
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", 
                   (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Active Persons: {self.current_person_count}", 
                   (info_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display_frame, f"Total Seen: {self.total_persons_seen}", 
                   (info_x, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Detections: {self.session_stats['total_detections']}", 
                   (info_x, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alert counters
        cv2.putText(display_frame, f"High Temp Alerts: {self.session_stats['high_temp_alerts']}", 
                   (info_x, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Emotion Alerts: {self.session_stats['emotion_alerts']}", 
                   (info_x, info_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Current alerts
        alert_y = 250
        for i, alert in enumerate(alerts[-3:]):  # Show last 3 alerts
            cv2.putText(display_frame, alert['message'], 
                       (10, alert_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return display_frame
    
    def export_session_data(self):
        """Export comprehensive session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare export data
        export_data = {
            'session_info': {
                'timestamp': timestamp,
                'total_persons_seen': self.total_persons_seen,
                'session_stats': self.session_stats
            },
            'persons': []
        }
        
        # Add current person data
        for person in self.active_persons.values():
            person_data = {
                'id': person.id,
                'first_seen': person.first_seen.isoformat(),
                'last_seen': person.last_seen.isoformat(),
                'frame_count': person.frame_count,
                'average_emotion': person.get_average_emotion(),
                'average_confidence': person.get_average_confidence(),
                'average_temperature': person.get_average_temperature(),
                'emotion_history': list(person.emotion_history),
                'temperature_history': list(person.temperature_history)
            }
            export_data['persons'].append(person_data)
        
        # Save to file
        filename = f"multi_person_session_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Session data exported: {filename}")
        return filename
    
    def run(self, camera_source=0):
        """Run the multi-person thermal monitoring system"""
        print("\n🚀 Starting Multi-Person Thermal Monitoring...")
        print("👥 Optimized for crowd screening and monitoring")
        print("\n🎮 Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'e' - Export session data")
        print("  'r' - Reset statistics")
        print("  SPACE - Pause/Resume")
        print("  'a' - Show/Hide alerts")
        print("\n" + "="*60)
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        show_alerts = True
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Could not read frame")
                        break
                    
                    # Calculate FPS
                    self.frame_count += 1
                    if self.frame_count % 10 == 0:
                        curr_time = time.time()
                        self.fps = 10 / (curr_time - self.prev_time)
                        self.prev_time = curr_time
                    
                    # Process frame
                    thermal_frame, persons = self.detect_and_track_persons(frame)
                    alerts = self.check_alerts(persons) if show_alerts else []
                    
                    # Draw display
                    display_frame = self.draw_multi_person_display(thermal_frame, persons, alerts)
                else:
                    # Show paused message
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                               (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Display frame
                cv2.imshow('Multi-Person Thermal Monitoring System', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"multi_person_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('e'):
                    self.export_session_data()
                elif key == ord('r'):
                    # Reset statistics
                    self.session_stats = {
                        'high_temp_alerts': 0,
                        'low_temp_alerts': 0,
                        'emotion_alerts': 0,
                        'total_detections': 0
                    }
                    self.total_persons_seen = 0
                    print("🔄 Statistics reset")
                elif key == ord(' '):  # Space bar
                    paused = not paused
                    print("⏸️  Paused" if paused else "▶️  Resumed")
                elif key == ord('a'):
                    show_alerts = not show_alerts
                    print("🚨 Alerts enabled" if show_alerts else "🔇 Alerts disabled")
        
        except KeyboardInterrupt:
            print("\n⚠️  System interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final session summary
            print("\n📊 FINAL SESSION SUMMARY:")
            print("="*40)
            print(f"👥 Total persons detected: {self.total_persons_seen}")
            print(f"🔥 High temperature alerts: {self.session_stats['high_temp_alerts']}")
            print(f"🧠 Emotion alerts: {self.session_stats['emotion_alerts']}")
            print(f"📈 Total detections: {self.session_stats['total_detections']}")
            
            # Auto-export final data
            final_export = self.export_session_data()
            print(f"💾 Final data exported: {final_export}")
            print("\n✅ Multi-Person Thermal System closed")

def main():
    """Main function"""
    print("👥🌡️  MULTI-PERSON THERMAL MONITORING SYSTEM")
    print("="*60)
    print("🎯 Advanced simulation for crowd screening scenarios")
    print("🏥 Perfect for: Airports, Buildings, Health Screening")
    print("🧠 Features: Person Tracking, Emotion Analysis, Temperature Monitoring")
    print("="*60)
    
    # Initialize and run system
    system = MultiPersonThermalSystem()
    system.run()

if __name__ == "__main__":
    main()