#!/usr/bin/env python3
"""
Simple RTSP detection script using the project's gridsearch Keras model.

This script connects to an RTSP stream, loads a Keras .h5 model (default:
`thermal_emotion_model_gridsearch.h5` in the repo root), preprocesses frames
to the expected input size (default 128x128, 3 channels), runs predictions,
and optionally displays and/or saves an annotated video.

Usage examples:
  python scripts/run_rtsp_detection.py --rtsp "rtsp://admin:Admin12345@192.168.1.142:554/Streaming/Channels/102"

Options:
  --model PATH       Path to .h5 model file (default: repo root gridsearch model)
  --rtsp URL         RTSP URL to open (required)
  --size N           Resize (W,H) target, default 128 (will use 128x128)
  --labels PATH      Optional labels file (one label per line). If missing,
                     defaults to the project's folder labels: angry, happy, natural, sad, surpise
  --display          Show annotated frames in a window (default: true)
  --save PATH        Optional path to save annotated video (mp4)
  --confidence C     Only show/publish predictions above C (float 0-1). Default 0.0

Notes/assumptions:
  - From repository documentation and notebook, the training pipeline used 128x128 RGB images
    and rescaled pixel values by 1./255. This script follows the same preprocessing.
  - The default label ordering assumes class indices from directory order (alphabetical):
    ['angry','happy','natural','sad','surpise']
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    # Use tensorflow.keras for model loading
    from tensorflow.keras.models import load_model
except Exception as e:
    print("Error importing tensorflow.keras. Make sure TensorFlow is installed in your environment.")
    raise


DEFAULT_LABELS = ["angry", "happy", "natural", "sad", "surpise"]


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
    # convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img


def annotate_frame(frame, text, pos=(10, 30), color=(0, 255, 0)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
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

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"Unable to open RTSP stream: {args.rtsp}")
        sys.exit(1)

    out_writer = None
    if args.save:
        # determine frame size from capture
        ret, frame = cap.read()
        if not ret:
            print("Unable to read a frame from stream to initialize writer")
            cap.release()
            sys.exit(1)
        h0, w0 = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        # use mp4v by default
        out_writer = open_video_writer(args.save, "mp4v", fps, (w0, h0))
        print(f"Saving annotated video to: {args.save}")
        # put the first frame back for processing loop by resetting capture (best effort)
        cap.release()
        cap = cv2.VideoCapture(args.rtsp)

    print("Starting RTSP processing. Press Ctrl+C or 'q' in the display window to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, retrying in 0.5s...")
                time.sleep(0.5)
                continue

            img = preprocess_frame(frame, target_size)
            # ensure correct channels
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 1:
                img = np.concatenate([img, img, img], axis=-1)

            x = np.expand_dims(img, axis=0)
            preds = model.predict(x)
            if preds.ndim == 2 and preds.shape[0] == 1:
                probs = preds[0]
            else:
                probs = np.ravel(preds)

            idx = int(np.argmax(probs))
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            conf = float(probs[idx])

            text = f"{label}: {conf:.2f}"
            if conf >= args.confidence:
                annotate_frame(frame, text)

            if args.display:
                cv2.imshow("RTSP Detection", frame)
                # allow window event handling
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
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
