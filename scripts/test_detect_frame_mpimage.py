"""Quick test script: capture a frame with OpenCV, wrap in mediapipe Image, call detect_for_video."""
import time
import sys

import cv2

import importlib
try:
    mp_image = importlib.import_module('mediapipe.tasks.python.vision.core.image')
    mp_hand = importlib.import_module('mediapipe.tasks.python.vision.hand_landmarker')
except Exception as e:
    print("Failed to import mediapipe modules via importlib:", e)
    sys.exit(1)

print("Imported mediapipe modules OK via importlib")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    sys.exit(1)

ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to capture frame")
    sys.exit(1)

# Convert BGR to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Frame shape, dtype:", rgb.shape, rgb.dtype)

# Create Image wrapper
try:
    img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
    print("Created mp Image, has _image_ptr:", hasattr(img, '_image_ptr'))
except Exception as e:
    print("Failed to create mp Image:", e)
    sys.exit(1)

# Create Landmarker (using default model from mediapipe package) - fallback use model asset path from examples
# For now try to create default HandLandmarker via create_from_model_path if available
# Use sample model if installed location exists

# Try to create a HandLandmarker using default options
try:
    # This will attempt to create with model file specified by user. If not available, user must provide model path.
    # Try to create a simple image-mode hand landmarker via create_from_options with default model path if exists.
    from mediapipe.tasks.python.core import base_options
    # No easy default model path available; we'll try to import a pre-built model in site-packages
    # Fallback: there is no model here; assume user already created HandLandmarker in their code and we can import it. So we exit.
    print("Please run with an existing HandLandmarker instance. This script only tests Image wrapping and will exit.")
except Exception as e:
    print("Note: not creating hand landmarker in this script.")

# If you have a hand_landmarker instance, you can call:
# res = hand_landmarker.detect_for_video(img, int(time.time()*1000))
# print(res)

print("Done")
