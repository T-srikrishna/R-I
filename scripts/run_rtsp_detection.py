#!/usr/bin/env python3
"""
RTSP detection script using the project's gridsearch Keras model.

Usage:
  python scripts/run_rtsp_detection.py --rtsp "rtsp://admin:Admin12345@192.168.1.142:554/Streaming/Channels/102"

Options:
  --model PATH       Path to .h5 model file (default: repo root gridsearch model)
  --rtsp URL         RTSP URL to open (required)
  --size N           Resize (W,H) target, default 128 (will use 128x128)
  --labels PATH      Optional labels file
  --display          Show annotated frames in a window (default: true)
  --save PATH        Optional path to save annotated video (mp4)
  --confidence C     Only show/publish predictions above C (float 0-1). Default 0.0

"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

try:
    # Use tensorflow.keras for model loading
    from tensorflow.keras.models import load_model
except Exception as e:
    print("Error importing tensorflow.keras. Make sure TensorFlow is installed in your environment.")
    raise


DEFAULT_LABELS = ["angry", "happy", "natural", "sad", "surpise"]


class EmotionTracker:
    """Track emotions over time to smooth predictions and reduce flickering."""
    def __init__(self, buffer_size=10, iou_threshold=0.5):
        self.tracks = {}  # track_id -> deque of (box, probs)
        self.next_id = 0
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        
    def compute_iou(self, box1, box2):
        """Compute Intersection over Union of two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of (x, y, w, h, label, conf, probs)
        Returns: list of smoothed (x, y, w, h, label, conf, probs)
        """
        # Match detections to existing tracks
        matched_tracks = set()
        updated_detections = []
        
        for det in detections:
            x, y, w, h, label, conf, probs = det
            box = (x, y, w, h)
            
            # Find best matching track
            best_iou = 0
            best_track_id = None
            
            for track_id, history in self.tracks.items():
                if len(history) > 0:
                    last_box, _ = history[-1]
                    iou = self.compute_iou(box, last_box)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            # Update existing track or create new one
            if best_track_id is not None:
                track_id = best_track_id
                matched_tracks.add(track_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = deque(maxlen=self.buffer_size)
            
            # Add to track
            self.tracks[track_id].append((box, probs))
            
            # Compute smoothed prediction
            smoothed_probs = self._smooth_predictions(track_id)
            smoothed_idx = int(np.argmax(smoothed_probs))
            smoothed_label = DEFAULT_LABELS[smoothed_idx] if smoothed_idx < len(DEFAULT_LABELS) else f"class_{smoothed_idx}"
            smoothed_conf = float(smoothed_probs[smoothed_idx])
            
            updated_detections.append((x, y, w, h, smoothed_label, smoothed_conf, smoothed_probs))
        
        # Remove old tracks that weren't matched
        tracks_to_remove = []
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return updated_detections
    
    def _smooth_predictions(self, track_id):
        """Average predictions over the track history."""
        history = self.tracks[track_id]
        if len(history) == 0:
            return np.zeros(len(DEFAULT_LABELS))
        
        # Average probabilities over time
        all_probs = [probs for _, probs in history]
        return np.mean(all_probs, axis=0)


def generate_output_filename(base_path=None):
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path:
        # If user provided a path, insert timestamp before extension
        path = Path(base_path)
        stem = path.stem
        suffix = path.suffix or ".mp4"
        parent = path.parent
        filename = f"{stem}_{timestamp}{suffix}"
        return str(parent / filename)
    else:
        # Default filename in current directory
        return f"thermal_detection_{timestamp}.mp4"


def load_labels_from_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        labels = [l.strip() for l in fh if l.strip()]
    return labels


def infer_model_io(model):
    # Try to get expected input size and channels from model.input_shape
    try:
        shape = model.input_shape  # typically (None, H, W, C)
    except Exception:
        return (128, 128, 3)

    if shape is None:
        return (128, 128, 3)

    # shape may be (None, h, w, c) or (None, c, h, w)
    if len(shape) == 4:
        _, a, b, c = shape
        # handle channels-first
        if a is None or b is None:
            # fallback
            return (128, 128, 3)
        if c in (1, 3):
            return (a, b, c)
        # if channels-first
        if c is None:
            # try other order
            _, c1, a1, b1 = shape
            if a1 and b1:
                return (a1, b1, c1)
    return (128, 128, 3)


def preprocess_frame(frame, target_size):
    # frame is a BGR image from cv2
    h, w = target_size
    img = cv2.resize(frame, (w, h))
    # convert BGR -> RGB (model was trained on RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # normalize to [0,1] like in training
    img = img.astype("float32") / 255.0
    return img


def detect_faces(frame, thresh_val=100, min_size=(50, 50), aspect_ratio_range=(0.75, 1.35), min_area=2500):
    """Enhanced thermal face detection using thresholding and contours.
    Returns list of (x,y,w,h) boxes.
    thresh_val: threshold applied on grayscale image (0-255)
    aspect_ratio_range: (min, max) acceptable width/height ratio for faces
    min_area: minimum contour area in pixels
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for better thermal detection
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    
    # Threshold
    _, th = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    h_img, w_img = frame.shape[:2]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Area filter (more strict)
        area = w * h
        if area < min_area:
            continue
        
        # Size filter
        if w < min_size[0] or h < min_size[1]:
            continue
        
        # Aspect ratio filter (faces are roughly square, shoulders/body are wide)
        aspect_ratio = w / float(h)
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
        
        # Position filter: faces are typically in upper 60% of frame
        # Skip detections in bottom 40% of frame (likely shoulders/body)
        if y > h_img * 0.6:
            continue
        
        # Size limit: faces shouldn't be too large (whole body)
        if w > w_img * 0.7 or h > h_img * 0.7:
            continue
        
        # Circularity check (faces are more circular than rectangular body parts)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Faces should have circularity > 0.4, body parts are more elongated
            if circularity < 0.35:
                continue
        
        # Expand a bit and clamp
        x1 = max(0, x - 8)
        y1 = max(0, y - 8)
        x2 = min(w_img, x + w + 8)
        y2 = min(h_img, y + h + 8)
        faces.append((x1, y1, x2 - x1, y2 - y1))
    
    return faces


def non_max_suppression(boxes, overlap_thresh=0.4):
    """Apply non-maximum suppression to bounding boxes."""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute the area of the bounding boxes
    area = boxes[:, 2] * boxes[:, 3]
    
    # Sort by area (larger first - prefer face over partial detections)
    idxs = np.argsort(area)[::-1]
    
    pick = []
    while len(idxs) > 0:
        # Pick the current box
        i = idxs[0]
        pick.append(i)
        
        # Find overlapping boxes
        suppress = [0]
        for pos in range(1, len(idxs)):
            j = idxs[pos]
            
            # Compute intersection
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            
            intersection = w * h
            
            # Compute overlap ratio
            overlap = intersection / float(min(area[i], area[j]))
            
            # If boxes overlap significantly, suppress the smaller one
            if overlap > overlap_thresh:
                suppress.append(pos)
        
        # Delete all suppressed boxes
        idxs = np.delete(idxs, suppress)
    
    return [tuple(boxes[i]) for i in pick]


def annotate_frame(frame, text, pos=(30, 60), color=(0, 255, 0)):
    # Draw a black background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(frame, (pos[0]-5, pos[1]-text_height-5), (pos[0]+text_width+5, pos[1]+5), (0,0,0), -1)
    # Draw text in larger font with thicker outline
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
    return frame


def open_video_writer(path, fourcc, fps, frame_size):
    # fourcc string like 'mp4v'
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(path, fourcc_code, fps, frame_size)


def main():
    parser = argparse.ArgumentParser(description="RTSP detection using the project's gridsearch Keras model")
    parser.add_argument("--rtsp", required=True, help="RTSP URL to open")
    parser.add_argument("--model", default=None, help="Path to .h5 model file (defaults to repo gridsearch model)")
    parser.add_argument("--size", default=128, type=int, help="Target size (square) to resize frames to. Default 128")
    parser.add_argument("--labels", default=None, help="Optional labels file (one label per line)")
    parser.add_argument("--display", action="store_true", help="Show annotated frames in a window")
    parser.add_argument("--save", default=None, help="Optional path to save annotated video (mp4 recommended)")
    parser.add_argument("--confidence", default=0.0, type=float, help="Only display results with confidence >= this")
    parser.add_argument("--skip", default=1, type=int, help="Process 1 of every N frames to reduce latency (default 1)")
    parser.add_argument("--thresh", default=120, type=int, help="Threshold value (0-255) for thermal contour detection (default 120)")
    parser.add_argument("--max-faces", default=2, type=int, help="Maximum number of face regions to evaluate per inference (default 2)")
    parser.add_argument("--nms-thresh", default=0.4, type=float, help="Non-maximum suppression overlap threshold (default 0.4)")
    parser.add_argument("--display-scale", default=1.0, type=float, help="Scale display window (e.g., 0.5 for half size, faster rendering, default: 1.0)")
    parser.add_argument("--min-face-size", default=50, type=int, help="Minimum face detection size in pixels (default 50)")
    parser.add_argument("--aspect-ratio-min", default=0.75, type=float, help="Minimum width/height ratio for face detection (default 0.75)")
    parser.add_argument("--aspect-ratio-max", default=1.35, type=float, help="Maximum width/height ratio for face detection (default 1.35)")
    parser.add_argument("--min-area", default=2500, type=int, help="Minimum contour area in pixels (default 2500)")
    parser.add_argument("--smoothing", default=10, type=int, help="Number of frames to smooth emotions over (default 10)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_model_path = repo_root / "thermal_emotion_model_gridsearch.h5"
    model_path = Path(args.model) if args.model else default_model_path

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # load labels
    if args.labels:
        labels = load_labels_from_file(args.labels)
    else:
        labels = DEFAULT_LABELS

    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path))
    input_h, input_w, input_c = infer_model_io(model)
    print(f"Model expects input approx: {input_w}x{input_h}x{input_c}")

    target_size = (args.size, args.size)
    # Prefer repository training size if it looks different
    if (input_h, input_w) != (args.size, args.size):
        target_size = (input_h, input_w)
        print(f"Using inferred model target size: {target_size}")

    # Initialize emotion tracker for temporal smoothing
    emotion_tracker = EmotionTracker(buffer_size=args.smoothing)

    # Open RTSP stream with optimizations for low latency
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    
    # CRITICAL: Set buffer size to 1 to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Unable to open RTSP stream: {args.rtsp}")
        sys.exit(1)
    
    print(f"RTSP stream opened successfully")
    
    # Test read a frame to verify stream is working
    ret, test_frame = cap.read()
    if not ret:
        print("WARNING: Unable to read initial test frame from stream")
    else:
        print(f"Stream resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
        del test_frame

    out_writer = None
    output_path = None
    if args.save:
        # Generate unique filename with timestamp
        output_path = generate_output_filename(args.save)
        print(f"Output will be saved to: {output_path}")
        
        # determine frame size from capture
        ret, frame = cap.read()
        if not ret:
            print("Unable to read a frame from stream to initialize writer")
            cap.release()
            sys.exit(1)
        h0, w0 = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        # use mp4v by default
        out_writer = open_video_writer(output_path, "mp4v", fps, (w0, h0))
        # put the first frame back for processing loop by resetting capture (best effort)
        cap.release()
        cap = cv2.VideoCapture(args.rtsp)

    print("Starting RTSP processing. Press Ctrl+C or 'q' in the display window to quit.")
    print(f"Face detection filters: min_size={args.min_face_size}px, aspect_ratio={args.aspect_ratio_min}-{args.aspect_ratio_max}, min_area={args.min_area}px²")
    print(f"Temporal smoothing: {args.smoothing} frames")
    
    try:
        frame_count = 0
        last_results = []  # reuse last detections when skipping frames
        skip_n = max(1, int(args.skip))
        thresh_val = int(args.thresh)
        last_time = time.time()
        fps_display = 0.0
        failed_reads = 0
        max_failed_reads = 50  # Exit after 50 consecutive failed reads
        
        while True:
            # LATENCY OPTIMIZATION: Use grab() to clear buffer and get latest frame
            for _ in range(3):  # Grab multiple times to skip buffered frames
                cap.grab()
            
            # Now retrieve the latest frame
            ret, frame = cap.retrieve()
            
            if not ret:
                failed_reads += 1
                if failed_reads >= max_failed_reads:
                    print(f"\nFailed to read {max_failed_reads} consecutive frames. Exiting.")
                    break
                # No frame available, wait a bit and retry
                time.sleep(0.01)
                continue
            
            # Successfully read a frame, reset failed counter
            failed_reads = 0
            frame_count += 1
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

            # Detect candidate face regions from thermal stream with enhanced filtering
            faces = detect_faces(
                frame, 
                thresh_val=thresh_val,
                min_size=(args.min_face_size, args.min_face_size),
                aspect_ratio_range=(args.aspect_ratio_min, args.aspect_ratio_max),
                min_area=args.min_area
            )
            
            # Apply non-maximum suppression to remove overlapping detections
            faces = non_max_suppression(faces, overlap_thresh=args.nms_thresh)
            
            # Debug: print face detection info occasionally
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}: Detected {len(faces)} face region(s) after filtering and NMS")

            # Only run model inference on 1 of every skip_n frames
            results = []
            if frame_count % skip_n == 0:
                # Sort faces by area (largest first) and cap to max_faces
                faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
                faces_to_eval = faces_sorted[: max(0, int(args.max_faces))]

                crops = []
                crop_boxes = []
                
                # Process detected faces (don't fall back to full frame)
                for (x, y, w, h) in faces_to_eval:
                    face = frame[y:y+h, x:x+w]
                    if face.size == 0:
                        continue
                    img = preprocess_frame(face, target_size)
                    # ensure 3 channels
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    if img.shape[-1] == 1:
                        img = np.concatenate([img, img, img], axis=-1)
                    crops.append(img)
                    crop_boxes.append((x, y, w, h))

                if crops:
                    X_batch = np.stack(crops, axis=0)
                    preds_batch = model.predict(X_batch, verbose=0)
                    # normalize preds_batch to 2D array if necessary
                    if preds_batch.ndim == 1:
                        preds_batch = np.expand_dims(preds_batch, axis=0)

                    for (box, probs) in zip(crop_boxes, preds_batch):
                        probs = np.ravel(probs)
                        idx = int(np.argmax(probs))
                        label = labels[idx] if idx < len(labels) else f"class_{idx}"
                        conf = float(probs[idx])
                        x, y, w, h = box
                        results.append((x, y, w, h, label, conf, probs))

                # Apply temporal smoothing to reduce flickering
                results = emotion_tracker.update(results)
                
                # update last_results for use on skipped frames
                last_results = results
            else:
                results = last_results

            # Annotate frame with results (if any)
            if results:
                for (x, y, w, h, label, conf, probs) in results:
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Main label
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    # Small prob list
                    for i, emo in enumerate(labels):
                        if probs[i] > args.confidence:
                            cv2.putText(frame, f"{emo}:{probs[i]:.2f}", (x, y+h+15 + i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            # FPS calc
            now = time.time()
            dt = now - last_time if last_time else 0.0
            if dt > 0:
                fps_display = 0.9*fps_display + 0.1*(1.0/dt)
            last_time = now
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Display and exit handling
            if args.display:
                # Scale display for faster rendering if requested
                display_frame = frame
                if args.display_scale != 1.0:
                    new_w = int(frame.shape[1] * args.display_scale)
                    new_h = int(frame.shape[0] * args.display_scale)
                    display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Thermal Emotion Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, exiting")
                    break

            if out_writer is not None:
                out_writer.write(frame)

    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
            if output_path:
                print(f"\nVideo saved to: {output_path}")
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()